#include "marian.h"

#include "translator/byte_array_util.h"
#include "translator/parser.h"
#include "translator/response.h"
#include "translator/response_options.h"
#include "translator/service.h"
#include "translator/utils.h"

#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  using namespace marian;
  
  CLI::App app{"Create a graphviz for a marian-fork model.", "graphviz"};

  std::string model = "";
  app.add_option("-m,--model", model, "The path to the model to visualize");

  CLI11_PARSE(app, argc, argv);
  
  if (model.empty()) {
    std::cerr << "A model must be provided" << std::endl;
    return 1;
  }
  
  YAML::Node config;
  std::stringstream configStr;
  marian::io::getYamlFromModel(config, "special:model.yml", model);
  configStr << config;
  
  auto graph = New<ExpressionGraph>();
  graph->setDevice(CPU0);
  graph->load(model);
  // Initialize the tensors.
  graph->forward();
  
  std::cout << graph->graphviz() << std::endl;
  
  return 0;
}
