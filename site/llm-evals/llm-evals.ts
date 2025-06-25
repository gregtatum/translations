export type ScoreNumbers = 1 | 2 | 3 | 4 | 5
export type Score = [ScoreNumbers, string]

export interface Evaluation {
  translation: Translation,
  scores: Scores | null,
}

export interface Translation {
  src: string,
  trg: string,
  ref: string,
}

export interface Scores {
  adequacy: Score,
  fluency: Score,
  terminology: Score,
  hallucination: Score,
  punctuation: Score
}

export interface Analysis {
  mean: number,
  median: number,
  histogram: Record<ScoreNumbers, number>
}

/**
 * Contains a human readable description of the term.
 */
interface Term {
  description: string,
  scales: Record<ScoreNumbers, string>
}

export interface Terminology {
  adequacy: Term,
  fluency: Term,
  terminology: Term,
  hallucination: Term,
  punctuation: Term
}
