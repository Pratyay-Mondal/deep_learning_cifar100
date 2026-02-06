# THWS/MAI/IDL25 Exam retake

**Magda Gregorova, 2025-11-30**

## Welcome to the IDL Final Project

This project simulates a real-world scenario you may encounter as a deep learning engineer: inheriting messy code, systematically debugging it, and optimizing models for production. Skills such as careful debugging, systematic experimentation, and clear communication of technical work are essential for success in the field.

**You shall work on the assignment individually!** (not in teams)

**Key to success**: Start early, as experimentation takes time. Document as you go - don't try to remember everything at the end. Be systematic and change one thing at a time. Be curious and understand *why* things work, not just *that* they work. **Ask for help early if you're stuck.**

### Submission instructions

The assignment is submitted via commits within this repo. The final commits for the two task in the assignment shall be tagged appropriately. Follow the instructions here below for more details. The state of the repo as of **Tuesday 20th January midnight** will be taken as decisive for the assignment - no later changes will reflected.

#### Oral presentation

Assignment presentation and discussion shall take place individually on **Friday 23rd of January**. Book your slot [here](https://terminplaner6.dfn.de/p/cd82ebc7632a1af25a30edcf23c1325c-1506266) - first come first serve basis.

#### Passing results

There are two tasks in the assignment and you need at least 60 points to pass.

### Academic integrity

This is an **individual assignment**. You may discuss high-level approaches with classmates (such as "I found data augmentation really helpful") but you may not share or copy code with them. 
**You may** use official PyTorch documentation and tutorials, reference course material, lectures, research papers, and internet sources including AI to understand concepts. **You may not** copy large blocks of code from tutorials, Stack Overflow, or AI tools without understanding them, or **submit code you don't understand!**

**The key principle**: Your submission should reflect **your understanding**. If you can't explain how your code works or why you made certain choices, that's a red flag.

### Good luck!

----

## Assignment Storyline - The Internship Crisis

You have been brought in as a Senior Deep Learning Engineer at a fast-paced tech startup. The company's critical product pipeline, which relies on a deep learning system built in-house in PyTorch, has been jeopardized by the work of a highly enthusiastic but ultimately inexperienced intern. While trying to improve the existing code and adapt it to a new dataset, the intern introduced several fundamental errors which caused that the code base no longer correctly operates even over the initial datasets. Unfortunately, the company as well as the intern have been also rather sloppy in version controlling and the initial well-operating version is nowhere to be found.

Your mission is twofold: **first, stabilize the codebase by identifying and fixing all critical bugs;** and **second, adapt and optimize the solution** to achieve production-ready performance on the new data. Success requires not just coding skill, but a rigorous, principled understanding of the deep learning pipeline you mastered in the IDL course.

---

## Task 1: Stabilize the Existing Codebase - Fix the Bugs (50 points)

You shall fix the errors and make sure that the code now operates correctly over the **CIFAR-10** dataset. The original code has been written by the manager and she has a strong preference for preserving it. ***Therefore she wants you to change the code as little as possible but make it work again and perform well.*** She also wants you to record all problems you discover, changes you introduce to fix these, and finally to explain these to her.

**Here is what she remembers about the code and would like to get back:** The MLP architecture should have 2 hidden layers with batchnorm, relu and dropout. The CNN architecture should have 2 blocks, each composed of Conv-BN-ReLU-Conv-BN-ReLU-MaxPool, followed by FC layers. The MLP shall reach over 50% accuracy on test data, and the CNN shall reach over 75% accuracy on test data.

**Info from the intern:** He didn't do any major changes, he only changed, added or deleted a few lines here and there. The code baseline is well structured and organized - he didn't mess it up, so you should not refactor the complete code. The errors are in individual lines of the code, and typically a one-line change is sufficient to fix the issue. There are altogether less than 10 issues in the code - luckily the intern was not in the company for so long.

### What you need to deliver for Task 1

1. **Code Repository (GitHub)**: Provide the final clean code with a complete, traceable commit history for all fixes and code changes. Use semantic commit messages (see https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716) and tag the final clean commit as `task1-final`, which will be used for testing.

2. **Complete Changelog**: Create a tabular listing (PDF or Markdown within the repo) of all code fixes documenting the commit SHA, commit message, change location (file and approximate lines), root cause or explanation of the problem, and the introduced change. ***You shall be able to explain how you discovered the problem - please make careful notes, "I do not remember" is not an acceptable answer!***

3. **Performance Report**: Provide validation of the final code (PDF or Markdown within the repo) showing training and testing performance with loss curves and evaluation metrics, including appropriate comments demonstrating that both models meet the performance targets.

---

## Task 2: Adapt and Optimize for Production (50 points)

With the codebase stabilized, the company now needs you to adapt the system for a new, more challenging dataset: **CIFAR-100**. 
The manager expects you to approach this task as a professional ML engineer: methodically test improvements, document your experiments, and deliver a well-optimized solution with clear justification for your design choices. For your final model, she expects at least 50% accuracy; over 65% would be excellent.

### Critical constraint: build on your fixed codebase

**You MUST start from your Task 1 stabilized code.** The manager wants to see how the existing architecture can be improved and adapted, not replaced with an entirely different solution downloaded from the internet. All improvements must be incremental modifications to your stabilized code.

**You are allowed** to add layers to the existing MLP or CNN architecture, implement residual connections by modifying your existing architecture, add or improve data augmentation in your data loading pipeline, modify the training loop to add features like LR scheduling or gradient clipping, tune hyperparameters such as learning rate, dropout rates, or weight decay, and add new training utilities like schedulers or better logging. However, **you are not allowed** to download pre-built ResNet/VGG/DenseNet implementations from GitHub or PyTorch Hub, replace your architecture with a completely different one from external sources, use pre-trained models or transfer learning, or copy-paste large blocks of code from external sources without understanding them.

**Why this constraint?** In industry, you often need to improve existing systems incrementally rather than rebuilding from scratch. This project teaches you to understand *why* certain techniques work and *how* to implement them yourself, not just how to download someone else's solution.

### What you need to deliver for Task 2

1. **Code Repository (GitHub)**: Provide the final clean code with a complete, traceable commit history for all code improvements and fixes. Update `readme.md` with all relevant information for the CIFAR-100 extension. Use semantic commit messages and tag the final clean commit as `task2-final`, which will be used for testing.

2. **Ablation Study**: Create a tabular listing (PDF or Markdown within the repo) of incremental improvements and the resulting performance. Document for each experiment the configuration with a detailed description of the implemented improvement (be specific), the hyperparameters used, the date and time when the experiment finished, train accuracy, test accuracy, training time, and key observations including whether the improvement delivered as expected and possible next steps. ***You shall be able to explain your reasons for trying these improvements - please make careful notes, "I do not remember" is not an acceptable answer! Loss plots or other validation metrics may be helpful for this.***

3. **Final Model Checkpoint** (link within readme): Save your best trained model weights and include a script to load and evaluate the model over the test dataset. Remember to make it reproducible to get the same results as in your ablation table. Describe the procedure and provide a download link to the checkpoint in `readme.md`.

***Remember: The journey of incremental improvements and careful analysis is at least as important as the final accuracy number!***


---

## Citation

AI (ChatGPT, Gemini and Claude) were used at various stages to get ideas, text and code snippets.

## License

The project is for educational purposes.