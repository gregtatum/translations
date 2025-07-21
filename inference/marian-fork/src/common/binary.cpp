#include "common/binary.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "common/io_item.h"
#include "common/types.h"
#include "tensors/cpu/integer_common.h"

#include <string>

namespace marian {
namespace io {

namespace binary {

struct Header {
  uint64_t nameLength;
  uint64_t type;
  uint64_t shapeLength;
  uint64_t dataLength;
};

// cast current void pointer to T pointer and move forward by num elements 
template <typename T>
const T* get(const void*& current, uint64_t num = 1) {
  const T* ptr = (const T*)current;
  current = (const T*)current + num;
  return ptr;
}

void loadItems(const void* ptrIntoModel, std::vector<io::Item>& items, bool mapped) {
  // Track total bytes loaded
  uint64_t totalBytesLoaded = 0;

  // Extract the binary file version.
  uint64_t binaryFileVersion = *get<uint64_t>(ptrIntoModel);
  ABORT_IF(binaryFileVersion != BINARY_FILE_VERSION,
           "Binary file versions do not match: {} (file) != {} (expected)",
           binaryFileVersion,
           BINARY_FILE_VERSION);
  totalBytesLoaded += sizeof(uint64_t);

  // Extract the number of headers.
  uint64_t numHeaders = *get<uint64_t>(ptrIntoModel);  
  totalBytesLoaded += sizeof(uint64_t);

  if(items.size() == numHeaders) {
    // The items have already been loaded, do not load them again.
    return;
  }

  // Extract the headers.
  const Header* headers = get<Header>(ptrIntoModel, numHeaders);
  totalBytesLoaded += sizeof(Header) * numHeaders;

  // Use the headers to initialize the std::vector<io::Item> items. This first loop will move
  // the model pointer forward as as it reads the names.
  items.resize(numHeaders);
  for(int i = 0; i < numHeaders; ++i) {
    items[i].mapped = mapped;
    items[i].type = (Type)headers[i].type;
    items[i].name = get<char>(ptrIntoModel, headers[i].nameLength);
    totalBytesLoaded += headers[i].nameLength;
  }

  // Extract the shapes, which will be copied to the tiems.
  for(int i = 0; i < numHeaders; ++i) {
    uint64_t len = headers[i].shapeLength;
    items[i].shape.resize(len);
    const int* shape = get<int>(ptrIntoModel, len);
    totalBytesLoaded += len * sizeof(int);
    std::copy(shape, shape + len, items[i].shape.begin());
  }

  // The model data is aligned to a 256 byte boundary. Move the model pointer forward.
  uint64_t offset = *get<uint64_t>(ptrIntoModel);
  totalBytesLoaded += sizeof(uint64_t);  // Account for offset metadata
  get<char>(ptrIntoModel, offset);
  totalBytesLoaded += offset;  // Account for offset bytes

  // Now load in the data for the items.
  for(int i = 0; i < numHeaders; ++i) {
    if (
      // Completely disable MMAP support for any models. We do not use it
      // in bergamot and we hijack this codepath for binary model loading.
      // If this is not set, we trigger node_initializers.cpp:186. This one
      // just assigns the memory ptr to the tensor if set to true, but at the moment
      // We are preparing some things on demand (the bottom portion of this code).
      // Once we stop doing that, we can use the full mmap codepath Also
      // when using the full mmap codepath, we need to uncomment expression_graph.h:582
      false
    ) {
      if(items[i].mapped && !isIntgemm(items[i].type)) {
        // memory-mapped, hence only set pointer. At the moment it intgemm matrices can't
        // be used without processing
        items[i].ptr = get<char>(ptrIntoModel, headers[i].dataLength);
      }
    } else {
      items[i].mapped = mapped;
    
      // Resize the item's owned data, and track how many bytes were loaded.
      auto resize = [&](uint64_t len) {
        items[i].bytes->resize(len);
        totalBytesLoaded += len;
      };
      
      const char* data = get<char>(ptrIntoModel, headers[i].dataLength);

      if(matchType<intgemm8>(items[i].type)) {
        if(items[i].name.find("Wemb") != std::string::npos) {
          // Dequantize the embedding weights from int8 to float32. This will allocate the
          // appropriate bytes on the io::Item's data vector.
          items[i].type = Type::float32;
          // We should have an extra float at the back but that requires a
          // different format, due to allocator work
          resize(items[i].shape.elements() * sizeof(float));  
          cpu::integer::unquantizeWemb<Type::int8>(items[i], data);
        } else {
          resize(headers[i].dataLength);
          cpu::integer::prepareAndTransposeB<Type::int8>(items[i], data);
        }
      } else if(matchType<intgemm16>(items[i].type)) {
        if(items[i].name.find("Wemb") != std::string::npos) {
          // Dequantize the embedding weights from int8 to float32. This will allocate the
          // appropriate bytes on the io::Item's data vector.
          items[i].type = Type::float32;
          // We should have an extra float at the back but that requires a
          // different format, due to allocator work
          resize(items[i].shape.elements() * sizeof(float));
          cpu::integer::unquantizeWemb<Type::int16>(items[i], data);
        } else {
          resize(headers[i].dataLength);
          cpu::integer::prepareAndTransposeB<Type::int16>(items[i], data);
        }
      } else {
        resize(headers[i].dataLength);
        std::copy(data, data + headers[i].dataLength, items[i].bytes->begin());
      }
    }
  }

  LOG(info, "[memory] Model data loaded in: {} bytes", totalBytesLoaded);
}

void loadItems(const std::string& fileName, std::vector<io::Item>& items) {
  // Read file into buffer
  uint64_t fileSize = filesystem::fileSize(fileName);
  std::vector<char> buf(fileSize);
// @TODO: check this again:
#if 1 // for some reason, the #else branch fails with "file not found" in the *read* operation (open succeeds)
  FILE *f = fopen(fileName.c_str(), "rb");
  ABORT_IF(f == nullptr, "Error {} ('{}') opening file '{}'", errno, strerror(errno), fileName);
  auto rc = fread(buf.data(), sizeof(*buf.data()), buf.size(), f);
  ABORT_IF(rc != buf.size(), "Error {} ('{}') reading file '{}'", errno, strerror(errno), fileName);
  fclose(f);
#else
  io::InputFileStream in(fileName);
  in.read(buf.data(), buf.size());
#endif

  // Load items from buffer without mapping
  LOG(info, "Loading model from file: {}", fileName);
  loadItems(buf.data(), items, false);
}

io::Item getItem(const void* modelPtr, const std::string& varName) {
  std::vector<io::Item> items;
  loadItems(modelPtr, items);

  for(auto& item : items)
    if(item.name == varName)
      return item;

  return io::Item();
}

io::Item getItem(const std::string& fileName, const std::string& varName) {
  std::vector<io::Item> items;
  loadItems(fileName, items);

  for(auto& item : items)
    if(item.name == varName)
      return item;

  return io::Item();
}

void saveItems(const std::string& fileName,
               const std::vector<io::Item>& items) {
  io::OutputFileStream out(fileName);
  uint64_t pos = 0;

  uint64_t binaryFileVersion = BINARY_FILE_VERSION;
  pos += out.write(&binaryFileVersion);

  std::vector<Header> headers;
  for(const auto& item : items) {
    headers.push_back(Header{item.name.size() + 1,
                             (uint64_t)item.type,
                             item.shape.size(),
                             item.bytes->size()}); // binary item size with padding, will be 256-byte-aligned
  }

  uint64_t headerSize = headers.size();
  pos += out.write(&headerSize);
  pos += out.write(headers.data(), headers.size());

  // Write out all names
  for(const auto& item : items) {
    pos += out.write(item.name.data(), item.name.size() + 1);
  }
  // Write out all shapes
  for(const auto& item : items) {
    pos += out.write(item.shape.data(), item.shape.size());
  }

  // align to next 256-byte boundary
  uint64_t nextpos = ((pos + sizeof(uint64_t)) / 256 + 1) * 256;
  uint64_t offset = nextpos - pos - sizeof(uint64_t);

  pos += out.write(&offset);
  for(uint64_t i = 0; i < offset; i++) {
    char padding = 0;
    pos += out.write(&padding);
  }

  // Write out all values
  for(const auto& item : items)
    pos += out.write(item.data(), item.bytes->size()); // writes out data with padding, keeps 256-byte boundary. 
                                                       // Amazingly this is binary-compatible with V1 and aligned and 
                                                       // non-aligned models can be read with the same procedure.
                                                       // No version-bump required. Gets 5-8% of speed back when mmapped.
}

}  // namespace binary
}  // namespace io
}  // namespace marian
