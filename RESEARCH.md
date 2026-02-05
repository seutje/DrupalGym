# **Engineering Domain-Specific Intelligence: A Technical Framework for Fine-Tuning Qwen2.5-Coder and Ministral-3 for Drupal 11 Development**

The emergence of specialized large language models has transitioned from a general-purpose utility to a critical component of high-velocity software engineering. In the context of the Drupal 11 ecosystem, the requirement for localized, high-fidelity code generation is driven by the framework's increasing architectural complexity, particularly its mandatory transition to PHP 8.3 and Symfony 7..1 While foundational models like Gemini 3 Flash provide broad coverage of popular languages, they often lack the "recency" and domain-specific nuances required for Drupal 11’s latest APIs, such as the Access Policy API or the modernized Icon API..4 Fine-tuning open-weights models like Qwen2.5-Coder:7B and Ministral-3:8B offers a pathway for organizations to bridge this intelligence gap while maintaining data sovereignty and reducing latency..6

## **Comparative Analysis of Foundational Architectures for Code Domain Adaptation**

The selection of a base model for Drupal-centric fine-tuning involves a strategic evaluation of parameter density, context window management, and pre-training data composition. Qwen2.5-Coder:7B and Ministral-3:8B represent the apex of contemporary sub-10-billion parameter models, yet they utilize divergent architectural philosophies..8

Qwen2.5-Coder:7B is a decoder-only Transformer built upon the Qwen2.5 architecture, which was pre-trained on a massive corpus of 5.5 trillion tokens..10 This model is explicitly designed for code-specific tasks, incorporating specialized tokens to handle diverse programming structures and a Fill-In-the-Middle (FIM) training objective that makes it exceptionally proficient at code completion..10 Its performance on the HumanEval benchmark (88.4%) suggests a superior grasp of algorithmic syntax compared to many larger models..12 The model architecture features 28 layers, 28 attention heads for Query and 4 for Key-Value (using Grouped-Query Attention), and a hidden size of 3,584..10

Ministral-3:8B, by contrast, is a more versatile model released by Mistral AI, prioritizing a balance between technical proficiency and broad world knowledge..9 While its raw coding benchmarks like HumanEval (34.8%) are significantly lower than Qwen’s, its higher scores in MMLU and MATH suggest a more robust capability for high-level architectural reasoning and document interpretation..12 For a developer tasked with writing complex Drupal routing logic that requires understanding of both Symfony components and Drupal-specific hooks, the broader linguistic foundation of Ministral may prevent the "hallucination" of non-existent APIs that often plagues purely code-centric models..13

| Architectural Feature | Qwen2.5-Coder:7B | Ministral-3:8B |
| :---- | :---- | :---- |
| Parameter Count | 7.61 Billion | 8.0 Billion |
| Primary Attention Mechanism | Grouped-Query Attention (GQA) | Sliding Window Attention / GQA |
| Context Window Capacity | 128,000 Tokens | 262,144 Tokens |
| Training Tokens | 5.5 Trillion | Undisclosed (High Density) |
| Position Embeddings | Rotary (RoPE) | RoPE |
| Activation Function | SwiGLU | SwiGLU |
| Normalization | RMSNorm | RMSNorm |
| License | Apache 2.0 | Mistral Research License |

The decision between these models often hinges on the target context. Qwen2.5-Coder:7B is optimized for the "developer in the trenches" who needs real-time completion of complex PHP 8.3 attributes and entity queries..15 Ministral-3:8B is better suited for architectural assistants that must ingest thousands of lines of documentation to explain the rationale behind a specific design pattern..9

## **The Drupal 11 Technical Paradigm: Implications for Domain Tuning**

The transition from Drupal 10 to 11 is not merely a version increment but a foundational shift in the technology stack. A fine-tuning pipeline must prioritize data that reflects these changes to avoid the regression of the model toward outdated patterns..2 The mandatory requirement of PHP 8.3 introduces features such as readonly properties, strict typing, and most importantly, the replacement of DocBlock annotations with native PHP attributes for plugin discovery..1

Historically, Drupal used annotations (e.g., @Block, @Action) within comments to define metadata for plugins. In Drupal 11, the Plugin API has shifted toward attribute-based discovery..4 This is a critical distinction for a fine-tuned model; a model trained on historical Drupal 8 or 9 code will consistently generate deprecated annotations, leading to immediate failure in a Drupal 11 environment..2 Fine-tuning datasets must therefore be heavily weighted toward code utilizing \#, \#\[Action\], and other attribute-based implementations..18

| Drupal Component | Drupal 10 Standard | Drupal 11/PHP 8.3 Requirement |
| :---- | :---- | :---- |
| Plugin Metadata | DocBlock Annotations | PHP 8.3 Attributes |
| Dependency Injection | Procedural \\Drupal::service() | Constructor Injection / Autowiring |
| Symfony Version | Symfony 6.4 | Symfony 7.1+ |
| Javascript Logic | jQuery/jQuery UI | Modern Vanilla JS / HTMX |
| Component Structure | Scattered Templates/CSS | Single Directory Components (SDC) |
| Administrative UI | Navigation (Contrib) | New Navigation (Core) |

Furthermore, the integration of Symfony 7 components—specifically the HttpKernel, EventDispatcher, and Routing components—means the model must understand the latest signatures and behaviors of these underlying libraries..3 The "Drupal-Symfony Partnership" has evolved to a state where Drupal core classes frequently extend or customize Symfony’s internal mechanisms, requiring the model to have a "cross-framework" understanding of how a request is transformed into a response within the kernel..20

## **Data Acquisition and Engineering for Specialization**

The primary bottleneck in any fine-tuning endeavor is the curation of a high-quality, high-signal dataset. For Drupal 11, the data gathering strategy must be multi-modal, combining raw source code, documentation, and synthetic instruction pairs..22

### **Source Code Extraction and Normalization**

The first layer of the dataset is the Drupal Core repository and compatible contributed modules. High-quality code can be found in projects that have already adopted Drupal 11 standards..25 Automation tools can be utilized to scrape the drupal.org project pages, filtering for modules that explicitly support the ^11 version constraint in their composer.json files..27

Once extracted, the code must undergo a cleaning and normalization process. This involves stripping boilerplate licensing headers that do not contribute to functional intelligence and using regular expressions to collapse consecutive whitespace or standardize comment formats..30 Deduplication is performed using hashing techniques to ensure that common boilerplate—such as the standard info.yml structures or basic hook implementations—does not overwhelm the model's training weights, which would lead to a reduction in its creative problem-solving ability..33

### **Instruction Engineering and Supervised Fine-Tuning (SFT) Formats**

Raw code is generally insufficient for instruction tuning; it must be converted into pairs that follow a logical prompt-response structure. The Alpaca format is the industry standard for this task, consisting of an instruction (the task description), an input (the context or specific code snippet), and an output (the correct completion or implementation)..35

For Drupal, these instruction pairs should focus on common developer tasks:

1. **Instruction**: "Create a custom block that displays the current user's last login date."  
2. **Input**: "Target Drupal 11 with PHP 8.3 attributes."  
3. **Output**:andEntityTypeManagerInterface\`\]..18

| Data Category | Extraction Method | Training Utility |
| :---- | :---- | :---- |
| Core Library | Git Clone / core/lib | Service Container and API logic |
| Contrib Attributes | Regex filter on src/Plugin | Mastering the new Attribute syntax |
| API Change Notices | Scrape api.drupal.org | Reducing hallucinations of old methods |
| Security Advisories | Scrape drupal.org/security | Learning to avoid common vulnerabilities |
| Symfony 7 Docs | Scrape symfony.com/doc | Understanding underlying framework changes |

### **Synthetic Data Augmentation via Teacher Models**

Because Drupal 11 is relatively new, the volume of human-written, production-grade code for its newest features (like the Icon API or the Access Policy API) is limited..4 To mitigate this, practitioners employ a "Teacher-Student" training regime. A more capable model, such as Gemini 3 Flash, is used to generate thousands of synthetic code samples based on raw Drupal 11 documentation..23 These samples are then filtered for quality using static analysis tools like PHPStan before being added to the training set for the 7B/8B model..38 This process, often orchestrated via frameworks like Genstruct-7B, ensures the model is exposed to a density of modern patterns that it could not find in the wild..41

## **Optimization via Parameter-Efficient Fine-Tuning (PEFT) and QLoRA**

Full fine-tuning of a 7B-8B parameter model is computationally expensive, often requiring significant VRAM to store the model weights, gradients, and optimizer states..42 To achieve professional-grade results on consumer or prosumer hardware, the Quantized Low-Rank Adaptation (QLoRA) method is utilized..44

### **The Mathematical Mechanism of LoRA/QLoRA**

LoRA operates by freezing the original weights of the large language model and injecting trainable, low-rank matrices into the Transformer layers..44 During training, only these small matrices (![][image1] and ![][image2]) are updated. QLoRA advances this by quantizing the base model to 4-bit precision (NormalFloat 4\) while keeping the adapters in a higher precision..44 This reduces the VRAM requirement of a 7B model from 100GB+ for full tuning down to approximately 12-16GB, allowing for training on a single NVIDIA RTX 3090 or 4090..46

The training objective for coding assistants is often enhanced by Fill-In-the-Middle (FIM) training..10 By presenting the model with a code prefix and a suffix and asking it to predict the "middle" content, the model learns to understand code not just as a linear stream but as a structured entity where the subsequent context influences the preceding logic..10

### **Implementation Frameworks: Unsloth, Axolotl, and TRL**

Choosing the right orchestration framework determines the efficiency of the training loop. Unsloth has emerged as the specialist for single-GPU optimization..49 By utilizing handwritten Triton kernels and manual backpropagation, Unsloth achieves 2-5x faster training speeds while reducing memory consumption by 70-80% compared to standard Hugging Face implementations..11

Axolotl is the preferred choice for developers seeking high customizability and multi-GPU support..51 It uses YAML-based configuration files, allowing for reproducible experiments where hyperparameters—such as the rank (![][image3]) of the adapters or the learning rate schedule—can be meticulously tuned..49

| Framework | Strength | Hardware Focus |
| :---- | :---- | :---- |
| Unsloth | Raw Speed and VRAM Efficiency | Single NVIDIA GPU (Consumer/Pro) |
| Axolotl | Flexibility and YAML Configs | Multi-GPU / Distributed setups |
| LLaMA Factory | No-code Web UI | Prototyping and Beginners |
| TRL (HF) | Integration with Ecosystem | General Research / Multi-node |
| Torchtune | PyTorch Native / No Abstractions | Advanced Researchers |

For a Drupal 11 project, Unsloth is the recommended starting point due to its native support for Qwen2.5-Coder and its ability to fit larger batch sizes into the 24GB VRAM limit of a prosumer GPU..8

## **Hardware Infrastructure and Computational Economics**

The hardware requirements for fine-tuning are strictly dictated by the model size and the desired context length..47 While inference can be performed on a wide range of devices including Apple Silicon or high-end CPUs, training remains firmly within the domain of NVIDIA GPUs due to the reliance on CUDA and specialized kernels..43

### **VRAM Requirements and GPU Selection**

For QLoRA fine-tuning of a 7B or 8B model with a standard sequence length of 2,048 tokens, 24GB of VRAM is the "sweet spot.".47 This makes the RTX 4090 the most cost-effective tool for independent developers or small agencies..47 If the goal is to fine-tune using a significantly longer context window (e.g., 32,768 tokens) to allow the model to ingest entire module directories at once, memory requirements increase significantly..11 In these scenarios, the use of A100 (80GB) or H100 GPUs is necessary to avoid Out-of-Memory (OOM) errors during the attention computation..47

| Model Size | Method | Min VRAM | Recommended Hardware |
| :---- | :---- | :---- | :---- |
| 7B \- 8B | QLoRA | 12-16 GB | RTX 3090 / RTX 4080 |
| 7B \- 8B | LoRA | 24-32 GB | RTX 4090 / A10 |
| 7B \- 8B | Full Fine-Tune | 100-120 GB | 2x A100 (80GB) |
| 14B \- 32B | QLoRA | 24-40 GB | A6000 / A100 (40GB) |
| 70B+ | QLoRA | 60-100 GB | 2x A100 or 1x H100 |

The A100 80GB provides a 46% throughput advantage over the RTX 3090 in high-concurrency environments, making it more suitable for multi-user production training where several LoRA adapters are being trained in parallel..54 For personal development or a single-project fine-tune, the performance difference is negligible compared to the significant cost savings of consumer hardware..54

### **Time and Resource Estimates**

Training time is a function of the dataset size and the number of training iterations (epochs)..47 For a 7B model like Qwen2.5-Coder on an RTX 4090, training on 10,000 instruction pairs typically takes 4 to 8 hours..47

| Training Phase | Dataset Size | hardware | Est. Duration |
| :---- | :---- | :---- | :---- |
| Prototyping (SFT) | 1,000 samples | RTX 4090 | 1 \- 2 Hours |
| Standard Run (SFT) | 10,000 samples | RTX 4090 | 4 \- 8 Hours |
| Deep Domain Tuning | 50,000 samples | RTX 4090 | 24 \- 36 Hours |
| Multi-Task Tuning | 100,000 samples | 2x A100 | 12 \- 18 Hours |

These estimates do not include the data preparation phase, which is often the most time-consuming part of the pipeline. Scrapping, cleaning, and instruction generation for a 10,000-sample dataset can take 2 to 4 weeks of engineering effort if started from scratch..7

## **The Optimal Pipeline: From Raw Data to Deployment**

The most efficient pipeline for a Drupal 11 development assistant follows a structured sequence of curation, training, and rigorous evaluation..7

### **Step 1: Curation and Deduplication**

Using a custom script, the engineering team clones the Drupal Core repository and a selection of top-tier contributed modules..25 The code is filtered for PHP 8.3 and Drupal 11 compatibility..27 Deduplication is performed to remove identical class structures, ensuring that the model learns the unique logic of each module..30

### **Step 2: Instruction Generation**

Using a teacher model (e.g., Gemini 3 Flash), the team generates instruction-response pairs for the gathered code..23 This step is critical for teaching the model how to follow developer intent..56 For example, a code block for a custom Entity Type is paired with an instruction like: "Define a new content entity type in Drupal 11 with support for revisions and translations.".4

### **Step 3: QLoRA Training with Unsloth**

The Qwen2.5-Coder:7B model is loaded into the Unsloth framework on an RTX 4090..8 The LoRA rank (![][image3]) is set to 32 to capture sufficient architectural detail, and the learning rate is configured to ![][image4] for stable convergence..35 The model is trained for 1 to 3 epochs, depending on the loss curve stabilization..8

### **Step 4: Quantization and Export**

Once training is complete, the LoRA adapters are merged with the base model, or exported as a standalone adapter file..35 To make the model usable in daily development, it is quantized to GGUF or EXL2 format..36 GGUF is particularly useful as it allows the model to run efficiently on CPU or mixed GPU/CPU environments via tools like Ollama or LM Studio..36

### **Step 5: IDE Integration**

The quantized model is integrated into the developer's IDE using extensions like Continue.dev or Tabby..36 This allows the model to act as a real-time copilot, providing code completions and architectural advice directly in the editor..7

## **Evaluation Metrics and Quality Assurance for Drupal Code**

The evaluation of a coding model must extend beyond general benchmarks to encompass the specific rules and standards of the Drupal ecosystem..58 A fine-tuned model that generates functionally correct code but violates Drupal's coding standards is a failure in a professional environment..39

### **Automated Standards Auditing**

The first line of defense is the Drupal Coder module, which provides rules for phpcs (PHP CodeSniffer)..39 The model's outputs are automatically piped through phpcs \--standard=Drupal to ensure adherence to whitespace rules, naming conventions, and DocBlock requirements..39 Furthermore, phpstan-drupal is used to perform static analysis, catching more subtle errors like the use of deprecated service calls or incorrect parameter types in entity queries..38

| Evaluation Tool | Check Type | Targeted Issue |
| :---- | :---- | :---- |
| PHPCS (Drupal) | Coding Standards | 2-space indents, file naming, snake\_case |
| PHPStan-Drupal | Static Analysis | Deprecated API calls, type mismatch |
| PHPUnit | Functional Testing | Logic errors in generated hooks/services |
| Gander | Performance Testing | Database query inefficiencies |
| Security Review | Security Audit | Direct SQL concatenation, XSS risks |

### **The 25-Point Security Scoring System**

Drupal uses a unique 25-point scale to score the severity of security vulnerabilities based on exploitability and impact..61 A fine-tuned assistant must be trained to recognize and avoid patterns that would result in a high security score..61 This includes ensuring that all user input is sanitized via the proper Render API mechanisms and that all database interactions use parameterized queries to prevent SQL injection..63 Evaluation sets should include intentionally vulnerable code for the model to identify and fix, demonstrating its "security-first" reasoning..64

### **Human-in-the-Loop Architectural Review**

For complex tasks—such as implementing a custom Access Policy—the model’s output is reviewed by senior Drupal architects..62 This "gold standard" evaluation measures whether the model followed the best architectural practices, such as using service-based approaches over procedural code and ensuring that all dependencies are properly injected rather than accessed through global helper functions..59

## **Future Outlook: Agentic Workflows and the Starshot Initiative**

The fine-tuning of 7B-8B models for Drupal 11 is not an isolated experiment but part of a broader shift toward "Agentic Coding.".7 With the launch of the Drupal Starshot initiative, which aims to make Drupal more accessible through a "Project Browser" and "Experience Builder," the demand for AI that can understand and generate "Drupal Recipes" will grow..2

Future iterations of this pipeline may involve training models not just to generate code, but to interact with the Drupal site itself via Drush..66 A model that can bootstrap a Drupal installation, run drush updb, and then analyze the resulting logs to fix compatibility errors represents the next level of developer productivity..67 As context windows expand and multi-modal capabilities are integrated into models like Ministral, the AI assistant will move from a simple code generator to a comprehensive technical partner capable of managing the entire lifecycle of a Drupal 11 application..2

## **Conclusions and Technical Recommendations**

To maximize the impact of a Drupal 11 fine-tuning project, organizations should adopt a pragmatic, iterative approach. The evidence suggests that Qwen2.5-Coder:7B currently holds a technical edge in raw code generation, while the Unsloth framework provides the most efficient computational path for small-to-mid-sized teams..8

The primary focus of engineering effort should remain on data quality rather than training duration. A model trained on 5,000 perfectly formatted, security-audited Drupal 11 instruction pairs will consistently outperform a model trained on 50,000 samples of noisy, outdated code..47 By integrating static analysis and security auditing directly into the evaluation loop, teams can ensure that their custom AI assistant is not just a tool for generating code, but a guardian of coding standards and architectural integrity in the Drupal 11 era..40

#### **Geciteerd werk**

1. \[11.x\] \[policy\] Require PHP 8.3 for Drupal 11, geopend op februari 5, 2026, [https://www.drupal.org/project/drupal/issues/3330874](https://www.drupal.org/project/drupal/issues/3330874)  
2. Drupal 11: What to expect? Comprehensive guide to new features ..., geopend op februari 5, 2026, [https://www.bulcode.com/insights/blog/drupal-11-what-expect-comprehensive-guide-new-features-and-enhancements](https://www.bulcode.com/insights/blog/drupal-11-what-expect-comprehensive-guide-new-features-and-enhancements)  
3. Symfony \- Drupalize.Me, geopend op februari 5, 2026, [https://drupalize.me/topic/symfony](https://drupalize.me/topic/symfony)  
4. Drupal APIs | Develop | Drupal Wiki guide on Drupal.org, geopend op februari 5, 2026, [https://www.drupal.org/docs/develop/drupal-apis](https://www.drupal.org/docs/develop/drupal-apis)  
5. Nichebench \- Benching AIs vs Drupal 10-11 \- Sergiu Nagailic, geopend op februari 5, 2026, [https://nikro.me/articles/professional/nichebench-benching-ais-vs-drupal-10-11/](https://nikro.me/articles/professional/nichebench-benching-ais-vs-drupal-10-11/)  
6. Automating Drupal Code Refactoring and Reviews with LLMs, geopend op februari 5, 2026, [https://www.bounteous.com/insights/2025/07/07/automating-drupal-code-refactoring-and-reviews-llms/](https://www.bounteous.com/insights/2025/07/07/automating-drupal-code-refactoring-and-reviews-llms/)  
7. Training a Large Language Model for Code (Code LLM) \- Drupal, geopend op februari 5, 2026, [https://www.drupal.org/project/artificial\_intelligence\_initiative/issues/3349218](https://www.drupal.org/project/artificial_intelligence_initiative/issues/3349218)  
8. Training, evaluation, compression, and deployment of a Qwen2.5, geopend op februari 5, 2026, [https://www.alibabacloud.com/help/en/pai/use-cases/training-evaluation-compression-and-deployment-of-qwen2-5-coder-model](https://www.alibabacloud.com/help/en/pai/use-cases/training-evaluation-compression-and-deployment-of-qwen2-5-coder-model)  
9. Ministral 3 8B 2512 vs. Qwen2.5 Coder 7B Instruct \- Galaxy.ai Blog, geopend op februari 5, 2026, [https://blog.galaxy.ai/compare/ministral-8b-2512-vs-qwen2-5-coder-7b-instruct](https://blog.galaxy.ai/compare/ministral-8b-2512-vs-qwen2-5-coder-7b-instruct)  
10. Qwen2.5-Coder Technical Report \- arXiv, geopend op februari 5, 2026, [https://arxiv.org/html/2409.12186v1](https://arxiv.org/html/2409.12186v1)  
11. unsloth/Qwen2.5-Coder-7B \- Hugging Face, geopend op februari 5, 2026, [https://huggingface.co/unsloth/Qwen2.5-Coder-7B](https://huggingface.co/unsloth/Qwen2.5-Coder-7B)  
12. Ministral 8B Instruct vs Qwen2.5-Coder 7B Instruct \- LLM Stats, geopend op februari 5, 2026, [https://llm-stats.com/models/compare/ministral-8b-instruct-2410-vs-qwen-2.5-coder-7b-instruct](https://llm-stats.com/models/compare/ministral-8b-instruct-2410-vs-qwen-2.5-coder-7b-instruct)  
13. Looks like not as good as Qwen2.5 7B \- Hugging Face, geopend op februari 5, 2026, [https://huggingface.co/mistralai/Ministral-8B-Instruct-2410/discussions/5](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410/discussions/5)  
14. Qwen2.5-Coder: Advanced Open-Source Code LLM \- Emergent Mind, geopend op februari 5, 2026, [https://www.emergentmind.com/topics/qwen2-5-coder](https://www.emergentmind.com/topics/qwen2-5-coder)  
15. unsloth/Qwen2.5-Coder-7B-Instruct Free Chat Online \- Skywork.ai, geopend op februari 5, 2026, [https://skywork.ai/blog/models/unsloth-qwen2-5-coder-7b-instruct-free-chat-online-skywork-ai/](https://skywork.ai/blog/models/unsloth-qwen2-5-coder-7b-instruct-free-chat-online-skywork-ai/)  
16. How to Upgrade your site from Drupal 9/10 to Drupal 11 | Source369, geopend op februari 5, 2026, [https://www.source369.com/insights/how-to-upgrade-your-site-from-drupal10-to-drupal11](https://www.source369.com/insights/how-to-upgrade-your-site-from-drupal10-to-drupal11)  
17. Roadmap to modernize code for D11 \+ PHP 8.3 \[\#3480491\] \- Drupal, geopend op februari 5, 2026, [https://www.drupal.org/project/recurring\_events/issues/3480491](https://www.drupal.org/project/recurring_events/issues/3480491)  
18. Exploring the Power of PHP Attributes in Drupal Development, geopend op februari 5, 2026, [https://www.qed42.com/insights/exploring-the-power-of-php-attributes-in-drupal-development](https://www.qed42.com/insights/exploring-the-power-of-php-attributes-in-drupal-development)  
19. 6 Powerful Ways to Interact and Extend Drupal 11 Using Core API, geopend op februari 5, 2026, [https://www.bhimmu.com/drupal/6-powerful-ways-to-interact-and-extend-drupal-11](https://www.bhimmu.com/drupal/6-powerful-ways-to-interact-and-extend-drupal-11)  
20. Symfony for Drupal Developers for Drupal 8, 9, 10, and 11, geopend op februari 5, 2026, [https://drupalize.me/guide/symfony-drupal-developers](https://drupalize.me/guide/symfony-drupal-developers)  
21. Drupal Meets Symfony: A Match Made for API Innovation \- QTA Tech, geopend op februari 5, 2026, [https://qtatech.com/en/article/drupal-meets-symfony-match-made-api-innovation](https://qtatech.com/en/article/drupal-meets-symfony-match-made-api-innovation)  
22. Data Preparation for SFT and PEFT \- NVIDIA Documentation, geopend op februari 5, 2026, [https://docs.nvidia.com/nemo-framework/user-guide/24.07/llms/starcoder2/dataprep.html](https://docs.nvidia.com/nemo-framework/user-guide/24.07/llms/starcoder2/dataprep.html)  
23. How to Create Custom Instruction Datasets for LLM Fine-tuning, geopend op februari 5, 2026, [https://www.firecrawl.dev/blog/custom-instruction-datasets-llm-fine-tuning](https://www.firecrawl.dev/blog/custom-instruction-datasets-llm-fine-tuning)  
24. Synthetic Dataset Generation for LLM Evaluation \- Langfuse, geopend op februari 5, 2026, [https://langfuse.com/guides/cookbook/example\_synthetic\_datasets](https://langfuse.com/guides/cookbook/example_synthetic_datasets)  
25. Drupal Code structure \- BeFused, geopend op februari 5, 2026, [https://befused.com/drupal/code-structure/](https://befused.com/drupal/code-structure/)  
26. GitLab CI templates will make Drupal 11 the default version to run, geopend op februari 5, 2026, [https://www.drupal.org/drupalorg/blog/gitlab-ci-templates-will-make-drupal-11-the-default-version-to-run](https://www.drupal.org/drupalorg/blog/gitlab-ci-templates-will-make-drupal-11-the-default-version-to-run)  
27. Provide Drush 13.x compatibility to be fully compatible with Drupal 11, geopend op februari 5, 2026, [https://www.drupal.org/project/data\_structures/issues/3507304](https://www.drupal.org/project/data_structures/issues/3507304)  
28. Installing a module compatible with the current Drupal version, geopend op februari 5, 2026, [https://docs.acquia.com/resources/installing-module-compatible-current-drupal-version](https://docs.acquia.com/resources/installing-module-compatible-current-drupal-version)  
29. Drupal 11 compatibility for Features module \[\#3447460\] | Drupal.org, geopend op februari 5, 2026, [https://www.drupal.org/project/features/issues/3447460](https://www.drupal.org/project/features/issues/3447460)  
30. PHP String Deduplication and Manipulation: Expert Techniques, geopend op februari 5, 2026, [https://medium.com/@kamrankhalid06/php-string-deduplication-and-manipulation-expert-techniques-b6bac23903e3](https://medium.com/@kamrankhalid06/php-string-deduplication-and-manipulation-expert-techniques-b6bac23903e3)  
31. Chat data cleaning, filtering and deduplication pipeline. \- GitHub, geopend op februari 5, 2026, [https://github.com/AlekseyKorshuk/chat-data-pipeline](https://github.com/AlekseyKorshuk/chat-data-pipeline)  
32. Preprocessing PHP to remove functionality from built files, geopend op februari 5, 2026, [https://stackoverflow.com/questions/6153412/preprocessing-php-to-remove-functionality-from-built-files](https://stackoverflow.com/questions/6153412/preprocessing-php-to-remove-functionality-from-built-files)  
33. Performance and Scalability of Data Cleaning and Preprocessing ..., geopend op februari 5, 2026, [https://www.mdpi.com/2306-5729/10/5/68](https://www.mdpi.com/2306-5729/10/5/68)  
34. Data cleaning and preprocessing \- Medium, geopend op februari 5, 2026, [https://medium.com/@sahilbansal480/data-cleaning-and-preprocessing-9680b71e00c3](https://medium.com/@sahilbansal480/data-cleaning-and-preprocessing-9680b71e00c3)  
35. Qwen 2.5 Coder 7B Base \+ 2x faster finetuning.ipynb \- Colab \- Google, geopend op februari 5, 2026, [https://colab.research.google.com/drive/1nOnpNubkGL5lZhKUBkFOWE5UajzieqCD?usp=sharing](https://colab.research.google.com/drive/1nOnpNubkGL5lZhKUBkFOWE5UajzieqCD?usp=sharing)  
36. A custom autocomplete model in 30 minutes using Unsloth ..., geopend op februari 5, 2026, [https://blog.continue.dev/a-custom-autocomplete-model-in-30-minutes-using-unsloth/](https://blog.continue.dev/a-custom-autocomplete-model-in-30-minutes-using-unsloth/)  
37. How to Create Synthetic Dataset EASILY? Step by Step Tutorial, geopend op februari 5, 2026, [https://m.youtube.com/watch?v=FAdRMVAWiak\&pp=0gcJCc0JAYcqIYzv](https://m.youtube.com/watch?v=FAdRMVAWiak&pp=0gcJCc0JAYcqIYzv)  
38. Configure PHPCS and PHPStan in DDEV for Drupal \- Eduardo Telaya, geopend op februari 5, 2026, [https://eduardotelaya.com/blog/technology/2025-07-21-configure-phpcs-and-phpstan-in-ddev-for-drupal/](https://eduardotelaya.com/blog/technology/2025-07-21-configure-phpcs-and-phpstan-in-ddev-for-drupal/)  
39. drupal/coder \- Packagist.org, geopend op februari 5, 2026, [https://packagist.org/packages/drupal/coder](https://packagist.org/packages/drupal/coder)  
40. mglaman/phpstan-drupal: Extension for PHPStan to allow ... \- GitHub, geopend op februari 5, 2026, [https://github.com/mglaman/phpstan-drupal](https://github.com/mglaman/phpstan-drupal)  
41. Generate Synthetic Instruction Dataset to Finetune LLMs \- Lightning AI, geopend op februari 5, 2026, [https://lightning.ai/lightning-ai/studios/generate-synthetic-instruction-dataset-to-finetune-llms](https://lightning.ai/lightning-ai/studios/generate-synthetic-instruction-dataset-to-finetune-llms)  
42. Qwen 3 8B vs Qwen 2.5 7B: Key Differences Explained \- Novita, geopend op februari 5, 2026, [https://blogs.novita.ai/qwen-3-and-qwen-2-5/](https://blogs.novita.ai/qwen-3-and-qwen-2-5/)  
43. QLoRA Fine-Tuning with Unsloth: A Complete Guide \- Medium, geopend op februari 5, 2026, [https://medium.com/@matteo28/qlora-fine-tuning-with-unsloth-a-complete-guide-8652c9c7edb3](https://medium.com/@matteo28/qlora-fine-tuning-with-unsloth-a-complete-guide-8652c9c7edb3)  
44. Your Local Coding Assistant: Fine-Tuning Big Models on a Budget ..., geopend op februari 5, 2026, [https://medium.com/@waleed.physics/your-local-coding-assistant-fine-tuning-big-models-on-a-budget-qlora-edition-6ed91e25edd8](https://medium.com/@waleed.physics/your-local-coding-assistant-fine-tuning-big-models-on-a-budget-qlora-edition-6ed91e25edd8)  
45. QLoRA vs LoRA: Which Fine‑Tuning Wins? | newline, geopend op februari 5, 2026, [https://www.newline.co/@Dipen/qlora-vs-lora-which-finetuning-wins--683ca660](https://www.newline.co/@Dipen/qlora-vs-lora-which-finetuning-wins--683ca660)  
46. QLoRA Memory Requirements \- LLM Fine-Tuning \- Repovive, geopend op februari 5, 2026, [https://repovive.com/roadmaps/llm-fine-tuning/lora-qlora-fundamentals/qlora-memory-requirements](https://repovive.com/roadmaps/llm-fine-tuning/lora-qlora-fundamentals/qlora-memory-requirements)  
47. Fine-Tuning Infrastructure: LoRA, QLoRA, and PEFT at Scale | Introl ..., geopend op februari 5, 2026, [https://introl.com/blog/fine-tuning-infrastructure-lora-qlora-peft-scale-guide-2025](https://introl.com/blog/fine-tuning-infrastructure-lora-qlora-peft-scale-guide-2025)  
48. \[Feature request\] Finetuning script for Qwen2.5-Coder FIM · Issue \#40, geopend op februari 5, 2026, [https://github.com/ggml-org/llama.vscode/issues/40](https://github.com/ggml-org/llama.vscode/issues/40)  
49. Unsloth AI: A Deep Dive into Faster, More Efficient LLM Fine-Tuning, geopend op februari 5, 2026, [https://skywork.ai/skypage/ko/Unsloth%20AI%3A%20A%20Deep%20Dive%20into%20Faster%2C%20More%20Efficient%20LLM%20Fine-Tuning/1972856091659923456](https://skywork.ai/skypage/ko/Unsloth%20AI%3A%20A%20Deep%20Dive%20into%20Faster%2C%20More%20Efficient%20LLM%20Fine-Tuning/1972856091659923456)  
50. Best frameworks for fine-tuning LLMs in 2025 \- Modal, geopend op februari 5, 2026, [https://modal.com/blog/fine-tuning-llms](https://modal.com/blog/fine-tuning-llms)  
51. LLM fine-tuning | LLM Inference Handbook \- BentoML, geopend op februari 5, 2026, [https://bentoml.com/llm/getting-started/llm-fine-tuning](https://bentoml.com/llm/getting-started/llm-fine-tuning)  
52. axolotl vs unsloth \[performance and everything\] : r/LocalLLaMA, geopend op februari 5, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1mltobj/axolotl\_vs\_unsloth\_performance\_and\_everything/](https://www.reddit.com/r/LocalLLaMA/comments/1mltobj/axolotl_vs_unsloth_performance_and_everything/)  
53. RTX 3090 vs A100 PCIe \- GPU Benchmarks \- Runpod, geopend op februari 5, 2026, [https://www.runpod.io/gpu-compare/rtx-3090-vs-a100-pcie](https://www.runpod.io/gpu-compare/rtx-3090-vs-a100-pcie)  
54. A100 vs RTX 3090 \- GPU Benchmark Comparison | Trooper.AI, geopend op februari 5, 2026, [https://www.trooper.ai/benchmarks-compare-A100-with-RTX-3090](https://www.trooper.ai/benchmarks-compare-A100-with-RTX-3090)  
55. \[P\] I fine-tuned Qwen 2.5 Coder on a single repo and got a ... \- Reddit, geopend op februari 5, 2026, [https://www.reddit.com/r/MachineLearning/comments/1jdiafd/p\_i\_finetuned\_qwen\_25\_coder\_on\_a\_single\_repo\_and/](https://www.reddit.com/r/MachineLearning/comments/1jdiafd/p_i_finetuned_qwen_25_coder_on_a_single_repo_and/)  
56. Gemma Fine-tuning for Beginners with Huggingface \- Kaggle, geopend op februari 5, 2026, [https://www.kaggle.com/code/heidichoco/gemma-fine-tuning-for-beginners-with-huggingface](https://www.kaggle.com/code/heidichoco/gemma-fine-tuning-for-beginners-with-huggingface)  
57. How to Fine-tune an LLM Part 3: The HuggingFace Trainer \- Wandb, geopend op februari 5, 2026, [https://wandb.ai/capecape/alpaca\_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy](https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy)  
58. \[Obsolete\] PHP coding standards | \[Obsolete\] PHP | Drupal Wiki ..., geopend op februari 5, 2026, [https://www.drupal.org/docs/develop/standards/coding-standards](https://www.drupal.org/docs/develop/standards/coding-standards)  
59. Comprehensive Guide of Best Practices for Drupal Development, geopend op februari 5, 2026, [https://medium.com/@imma.infotech/comprehensive-guide-of-best-practices-for-drupal-development-e73d4ba64029](https://medium.com/@imma.infotech/comprehensive-guide-of-best-practices-for-drupal-development-e73d4ba64029)  
60. An enhanced coding standard for Drupal projects. \- GitHub, geopend op februari 5, 2026, [https://github.com/previousnext/coding-standard](https://github.com/previousnext/coding-standard)  
61. Drupal's Security: Best Practices | Pantheon.io, geopend op februari 5, 2026, [https://pantheon.io/learning-center/drupal-security](https://pantheon.io/learning-center/drupal-security)  
62. BEST PRACTICES | Drupal.org, geopend op februari 5, 2026, [https://new.drupal.org/case-study/best-practices](https://new.drupal.org/case-study/best-practices)  
63. Drupal Security: Best Practices in 2025 \- Wishdesk, geopend op februari 5, 2026, [https://wishdesk.com/blog/drupal-security-best-practices-in-2025](https://wishdesk.com/blog/drupal-security-best-practices-in-2025)  
64. AI Code Review for Drupal \- bPekker.dev, geopend op februari 5, 2026, [https://bpekker.dev/ai-code-review/](https://bpekker.dev/ai-code-review/)  
65. How to Write Effective Drupal Test Cases Easily \- August Infotech, geopend op februari 5, 2026, [https://www.augustinfotech.com/blogs/how-to-write-test-cases-for-drupal-project/](https://www.augustinfotech.com/blogs/how-to-write-test-cases-for-drupal-project/)  
66. Drupal 101 — Install Drush on Windows and troubleshooting \- Medium, geopend op februari 5, 2026, [https://medium.com/@kamiviolet/drupal-101-install-drush-on-windows-and-troubleshooti-3d26ff812112](https://medium.com/@kamiviolet/drupal-101-install-drush-on-windows-and-troubleshooti-3d26ff812112)  
67. Bootstrap \- Drush, geopend op februari 5, 2026, [https://www.drush.org/13.x/bootstrap/](https://www.drush.org/13.x/bootstrap/)  
68. Drush | Development tools | Drupal Wiki guide on Drupal.org, geopend op februari 5, 2026, [https://www.drupal.org/docs/develop/development-tools/drush](https://www.drupal.org/docs/develop/development-tools/drush)  
69. Automated Drupal 11 compatibility fixes for lightning\_core \+ enable ..., geopend op februari 5, 2026, [https://www.drupal.org/project/lightning\_core/issues/3369498](https://www.drupal.org/project/lightning_core/issues/3369498)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAXCAYAAADUUxW8AAAAsklEQVR4XmNgGNbAFYgfogsSC/4D8Vd0QWLABAaIZhAmCfAA8Wsg/s1AhubrQCzMAPEvSZoNgXg+lH2AAaIZ5BKiwD8gZoSylzJANEsipHGDciAOQuK3MkA0GyOJYQUsQPweTayIAaIZ2UCsAORcdACyEaR5IboEMgDFqS26IAPEryDNB9DE4YCfAdO5MAAKZZBmjCTqDcRfGBCp6D6qNMMFIP4LlQPhR0AcjqJiFAx2AAB+zydx0TuHpwAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAXCAYAAAAC9s/ZAAAAxUlEQVR4XmNgGAXooAmIHwHxfyh+BuWD8BMkcW6YBlwApOg3uiAQcDJA5P6hSyADkOkgRWvQJaAA5gqcwIUBosAGXYIB4YKf6BLIYCsDRBELugQQvGeAyPGjSyADkP/+ALEkEs5igGhch6QOK+BggCjczIBqgCoQ/wLiO0DMCFeNBeDzP8hLILmv6BLI4DkD/hAmGAMgyQfoglBgzgCRv4IuAQNKDBAF5egSQCDEgLAdIwy8GVCT7ysoH4Zh4sdhGkbBsAMAExc4rTUi4lkAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAkAAAAZCAYAAADjRwSLAAAAb0lEQVR4XmNgGAVUB7+B+AYQ3wJibyD+BsQHgHgPTEEEEOsAsSkQ/wfi+VBxEBuEweA5lJ6ELAgEv4A4C8YBGQ8Ch4H4H0wQFwCZshVdEBnwMEAUuaBLIAM/BogiDnQJZADyKkH3fALi6eiCowAOABeGF0NsowoUAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAXCAYAAABZPlLoAAAB5UlEQVR4Xu2Wv0oEMRDGP1ELURQtFEGwsRHBRtBOH8LCTuysrrIQ7H0JQSzV2kqwWPQZ7CzUQsFCUbRR/JNhEm5uLtns7d1uY34w7OZLliRfNpMAiUSEPhO/WkwwGZI5XlZMLCJiTgPcgOLbxExrda2MIn+wW2iO9cZEf0ttcWg7HZoYQU5/FyY2RPkS3HhdaFUzCV4UN+nQYA9MvIjyMrjtgNCKcm6fQXNcxY7S8wZYNXfw9z0E1ueV/mHiWJSlwb4gFkzM2fegOdSRzwg3wGmlaza1oKAJ6cnECJmzDdZpMpIrq3fCiYl7Gw/g7+l9WDYi9k2sKu0R/MGU0jVjJj61aKE62iqdEjKHtj/pegKZ1SmHlIF+AF9/Xty5/6MrAkyg3QRnTJkBh8wJ6RlYH1d6EXbR3Dm+v7KNI3DDWV2RgzSoG2OIkAkhPQPrsRTQNS77lznKnUHdGEOETAjpGWowh1acOqEkWoaqzcnAuj62nR7dEmUZBHcgL1SUqGMJ2VHHttqD34Qyp1VhaBJf9im5VeUQdSVkWijS9dXgzcS10noG3TipU1/EqOIof0K4b7qTnImySwVlcmSUJbQb4oLuOjF6dQl0t9R3NC9nz1aTiZb+xFfwgp7a+jVRn0gkEonEP+IP/9+d71QCn98AAAAASUVORK5CYII=>