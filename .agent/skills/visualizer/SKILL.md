---
name: mnist-viz-specialist
description: Use this skill to generate visual performance reports, confusion matrices, and sample digit galleries for MNIST models.
---

# Mission
You are an expert at evaluating MNIST classification. When this skill is active:
1. Generate a Matplotlib grid showing 10 random test images, their true labels, and the model's predicted labels.
2. Color code the labels: Green for correct, Red for incorrect.
3. If the user is only using 3 digits (0, 1, 2), ensure the confusion matrix only shows those 3 classes.
4. Save the result as `performance_artifact.png` and present it as a Walkthrough Artifact.

# Constraints
- Always use 'Agg' backend for Matplotlib to ensure it runs in the agent's background terminal.
- Use `sns.heatmap` for the confusion matrix if `seaborn` is available.