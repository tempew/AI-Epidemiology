# AI Epidemiology: Achieving Explainable AI Through Expert Oversight Patterns

**Kit Tempest-Walters**  
ktempestwalters@gmail.com

---

## Abstract

AI Epidemiology is a framework for governing and explaining advanced AI systems by applying population-level surveillance methods to AI outputs. The approach mirrors the way in which epidemiologists enable public health interventions through statistical evidence before molecular mechanisms are understood. This bypasses the problem of model complexity which plagues current interpretability methods (such as SHAP and mechanistic interpretability) at the scale of deployed models.

AI Epidemiology achieves this population-level surveillance by standardising capture of AI-expert interactions into structured assessment fields: risk level, alignment score, and accuracy score. These function as exposure variables which predict output failure through statistical associations, much like cholesterol and blood pressure act as exposure variables predicting cardiac events. Output-failure associations are subsequently validated against expert overrides and real-world outcomes.

The framework places zero burden on experts and provides automatic audit trails by passively tracking expert convergence and divergence with AI recommendations. Since it analyses outputs rather than internal model computations, it also provides governance continuity when institutions update models and switch vendors. Finally, by providing reliability scores and semantic assessments (e.g. 'this recommendation resembles 500 cases overridden by experts due to guideline violations'), it enables experts and institutions to detect unreliable AI outputs before they cause harm. This democratises AI oversight by enabling domain experts to govern AI systems without requiring machine learning expertise.

---

## Paper

📄 ** https://github.com/tempew/AI-Epidemiology/blob/main/AI%20Epidemiology.pdf (AI Epidemiology.pdf)**

- 41 pages, 4 tables
- Submitted to arXiv (cs.AI, pending moderation) and AI & Society
- November 2025

---

## Key Contributions

1. **Novel Framework**: Applies epidemiological surveillance methods to AI governance, bypassing intractable model interpretability challenges

2. **Logia Protocol**: Operationalises AI Epidemiology through standardized capture of AI-expert interactions with structured assessment fields

3. **Empirical Validation**: Feasibility study in ophthalmology demonstrates 94% inter-rater reliability (ICC = 0.89) for measurement standardisation

4. **Model-Agnostic Governance**: Provides governance continuity across model updates and vendor changes by analysing outputs rather than internal computations

---

## Core Components

### Logia Grammar
Structures AI-expert interactions into standardised fields:
- **Mission**: User query/instruction
- **Conclusion**: AI recommendation  
- **Justification**: AI reasoning
- **Assessment fields**: Risk level, alignment score, accuracy score
- **Expert actions**: Override decision, corrective option

### Tracelayer
Population-level pattern analysis enabling:
- Reliability scores for individual outputs
- Semantic explanations (e.g., "this output resembles 500 cases overridden by experts")
- Guidance for model correction and interpretability research

### Dual Stratification
- **Risk level**: Stratifies by consequence severity
- **Reliability score**: Predicts output failure probability

---

## Citation
```bibtex
@article{tempestwalters2025epidemiology,
  title={AI Epidemiology: Achieving Explainable AI Through Expert Oversight Patterns},
  author={Tempest-Walters, Kit},
  journal={arXiv preprint},
  year={2025},
  note={Pending moderation}
}
```

---

## Status

- Paper complete (41 pages)
- ArXiv submission pending endorsement
- Submitted to AI & Society
- Code and RAG documents available upon request

---

## Contact

For questions, collaboration, or access to implementation code:  
**ktempestwalters@gmail.com**

---

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
