# 2026-Spring-Brown-UTRA
My work during my Undergraduate Teaching and Research Awards term as a Machine Learning intern at Brown.

## Project Background
My research investigates the representation of memory expression in the brain. The current predominant belief in the episodic memory field is that remembering an event from the past requires the reactivation of cortical patterns initially evoked during perception (Danker & Anderson, 2010). However, accumulating evidence regarding how significant representational changes occur when percepts are transformed into memories is challenging this view (Breedlove et al., 2020; Favila et al., 2022; Liu et al., 2021). Currently, the underlying mechanisms of  the mechanism underlying this change, and its consequences for memory behavior, remain unknown. The broader goal of my project is to determine how representational changes between perception and memory emerge, and how these changes influence later behavior.

A major obstacle to solving this is that gauging human visual recall has historically been a challenge. Previous research often used recognition tests, or cued recall tasks that involved multiple choice questions. While these methods were easy to score, they often fail to capture the tendency of human memory to. To address this problem, my project will incorporate large language models(LLMS) to allow the testing of free recall. By training LLMs, such as Llama, we can use them as judges for patient descriptions and score their raw memory recall of a picture.  The question that my research addresses is what LLM models lead to the most accurate quantification of this data. 

## Methods
To gauge the effectiveness of models in comparison, our approach is to track the “within-image” cosine similarities as the layers progress. If at the deepest layer, layer 28 has a within image cosine similarity of .85 and across-image similarity of .30, this indicates the model has successfully learned to group different descriptions of the same scene together while distinguishing them from unrelated images.

The three models that were tested are 
1. Llama 3.2 (3B)
2. MPNet-base-v2
3. Sentence-Transformers (General)

There were three layers that were targetted:

Low: surface level features

Middle: semantic structures, basic relationships

High: high level abstractions, instruction-tuned logic

## Results
My results showed strong potential for AI automation of scoring since all three models successfully distinguished between captions of the same image versus different images. However, specialized sentence-embedding models (MPNet) showed stronger semantic discriminability, likely due to their contrastive learning objective which structures the embedding space to preserve semantic distinctions. The next step is to test whether this optimized semantic space better captures gist-level representations.

## Conclusion
By optimizing LLMs to score visual memory recall, we can move beyond simplified testing and gain further insight into the complex transformations that occur between perception and memory. 
