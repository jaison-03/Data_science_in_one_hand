export type ArchitectureLayer = {
  name: string;
  acronym: "AI" | "ML" | "DL" | "NLP";
  subtitle: string;
  radius: number;
  color: string;
  description: string;
  facts: string[];
};

export const architectureLayers: ArchitectureLayer[] = [
  {
    name: "Artificial Intelligence",
    acronym: "AI",
    subtitle: "Artificial Intelligence",
    radius: 3,
    color: "#83cfff",
    description: "The broadest field of intelligent computing.",
    facts: [
      "Broadest field of intelligent computing.",
      "Includes machine learning, deep learning, natural language processing, and other AI techniques.",
      "Enables machines to perform tasks that typically require human intelligence.",
    ],
  },
  {
    name: "Machine Learning",
    acronym: "ML",
    subtitle: "Machine Learning",
    radius: 2.35,
    color: "#82e0b0",
    description: "A subset of AI that learns useful patterns from data.",
    facts: [
      "A subset of artificial intelligence.",
      "Learns patterns from data.",
      "Uses algorithms to make predictions and decisions.",
    ],
  },
  {
    name: "Deep Learning",
    acronym: "DL",
    subtitle: "Deep Learning",
    radius: 1.65,
    color: "#ffc27d",
    description: "A subset of ML built around multilayer neural networks.",
    facts: [
      "A subset of machine learning.",
      "Uses multilayer neural networks.",
      "Excels at learning complex patterns from data.",
    ],
  },
  {
    name: "Natural Language Processing",
    acronym: "NLP",
    subtitle: "Natural Language Processing",
    radius: 0.88,
    color: "#c5a1ff",
    description: "A field focused on enabling computers to understand and process human language.",
    facts: [
      "Focuses on understanding and processing human language.",
      "Includes text analysis, translation, and summarization.",
      "Used in chatbots, virtual assistants, and language applications.",
    ],
  },
];
