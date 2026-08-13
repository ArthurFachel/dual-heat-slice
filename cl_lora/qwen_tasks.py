"""
qwen_tasks.py — Small controlled text tasks for LLM Continual Learning experiments.

Defines 4 text classification/generation tasks across different domains.
Each task has a small dataset (hundreds of examples) suitable for fast
experimentation on limited hardware.

Task domains:
  A — Sentiment classification (positive/negative movie reviews)
  B — Topic classification (sports vs. technology)
  C — Question type classification (yes/no vs. factual)
  D — Toxicity detection (toxic vs. safe)

Each task is a supervised text classification problem formatted as
instruction-following prompts compatible with causal LM fine-tuning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

from datasets import Dataset


# ── Data ─────────────────────────────────────────────────────────────────

SENTIMENT_TRAIN = [
    # Positive
    ("This movie was absolutely fantastic, I loved every minute of it.", "positive"),
    ("A wonderful experience that left me feeling inspired.", "positive"),
    ("The acting was superb and the plot kept me on the edge of my seat.", "positive"),
    ("One of the best films I have seen this year, highly recommended.", "positive"),
    ("Beautiful cinematography and a touching story.", "positive"),
    ("The characters were well-developed and the dialogue felt natural.", "positive"),
    ("An incredible journey from start to finish.", "positive"),
    ("I was pleasantly surprised by how good this turned out to be.", "positive"),
    ("A masterpiece of modern cinema.", "positive"),
    ("The director did an amazing job bringing this story to life.", "positive"),
    ("Thoroughly entertaining from beginning to end.", "positive"),
    ("A heartwarming story with great performances.", "positive"),
    ("The soundtrack alone makes this worth watching.", "positive"),
    ("Clever writing and excellent pacing throughout.", "positive"),
    ("This film exceeded all my expectations.", "positive"),
    ("A delightful comedy that had me laughing out loud.", "positive"),
    ("The visual effects were stunning and groundbreaking.", "positive"),
    ("An emotionally resonant film that stays with you.", "positive"),
    ("Brilliant performances from the entire cast.", "positive"),
    ("A must-see for fans of the genre.", "positive"),
    # Negative
    ("This was a complete waste of time, the plot made no sense.", "negative"),
    ("Terrible acting and even worse writing.", "negative"),
    ("I can not believe I sat through the entire thing.", "negative"),
    ("A boring and predictable mess from start to finish.", "negative"),
    ("The worst movie I have seen in years.", "negative"),
    ("Nothing about this film worked, it was painfully dull.", "negative"),
    ("The director should be embarrassed by this production.", "negative"),
    ("Awful pacing and completely unlikable characters.", "negative"),
    ("I regret spending money on this disaster of a film.", "negative"),
    ("The dialogue was cringe-worthy and the plot was incoherent.", "negative"),
    ("A painfully generic film with zero originality.", "negative"),
    ("This movie was a trainwreck from the first scene.", "negative"),
    ("Lazy writing and phoned-in performances.", "negative"),
    ("One of the most disappointing films I have ever seen.", "negative"),
    ("The special effects were laughably bad.", "negative"),
    ("I fell asleep halfway through, it was that boring.", "negative"),
    ("A shallow and poorly executed attempt at storytelling.", "negative"),
    ("The script was riddled with cliches and bad dialogue.", "negative"),
    ("An utterly forgettable film that wastes its cast.", "negative"),
    ("Save your time and skip this one entirely.", "negative"),

    ("A truly captivating film that keeps you guessing until the end.", "positive"),
    ("The performances were raw, emotional, and completely believable.", "positive"),
    ("An absolute triumph in storytelling and character development.", "positive"),
    ("This film is a gem, beautifully shot and expertly acted.", "positive"),
    ("Hilarious from start to finish, a comedy masterpiece.", "positive"),
    ("The depth of emotion in this film is remarkable.", "positive"),
    ("A visually stunning masterpiece with a powerful message.", "positive"),
    ("The cast delivered outstanding performances across the board.", "positive"),
    ("I was completely drawn into this film's world from the first scene.", "positive"),
    ("An extraordinary piece of cinema that deserves all the praise.", "positive"),
    ("The screenplay was sharp, witty, and perfectly paced.", "positive"),
    ("A refreshingly original film in a sea of sequels and remakes.", "positive"),
    ("The director's vision is realized beautifully on screen.", "positive"),
    ("This movie touched my heart in ways I did not expect.", "positive"),
    ("A brilliant blend of humor, heart, and breathtaking visuals.", "positive"),
    ("Every scene was crafted with care and attention to detail.", "positive"),
    ("One of those rare films that gets better with each viewing.", "positive"),
    ("The score and cinematography work in perfect harmony.", "positive"),
    ("An inspiring story told with grace and authenticity.", "positive"),
    ("This film left me speechless absolutely phenomenal.", "positive"),
    ("The pacing was so slow I almost walked out of the theater.", "negative"),
    ("A poorly written script that relies on cheap cliches.", "negative"),
    ("The acting was wooden and completely unconvincing.", "negative"),
    ("An incoherent mess of a plot that goes nowhere.", "negative"),
    ("This film is an insult to the intelligence of its audience.", "negative"),
    ("The special effects looked like they were from twenty years ago.", "negative"),
    ("Boring, predictable, and utterly forgettable.", "negative"),
    ("The characters were one-dimensional and unlikeable.", "negative"),
    ("A lazy cash grab with zero artistic merit.", "negative"),
    ("The dialogue was cringeworthy and the jokes fell flat.", "negative"),
    ("I kept checking my watch willing it to be over.", "negative"),
    ("The direction was amateurish and the editing was choppy.", "negative"),
    ("A complete disaster from start to finish.", "negative"),
    ("The soundtrack was jarring and completely out of place.", "negative"),
    ("Not even the talented cast could save this trainwreck.", "negative"),
    ("The plot holes were big enough to drive a truck through.", "negative"),
    ("This movie had no soul no heart no redeeming qualities.", "negative"),
    ("A pointless remake that nobody asked for.", "negative"),
    ("The film tries too hard to be profound but ends up pretentious.", "negative"),
    ("I want those two hours of my life back.", "negative"),]

TOPIC_TRAIN = [
    # Sports
    ("The quarterback threw a touchdown pass in the final minute.", "sports"),
    ("The basketball team won the championship after an incredible season.", "sports"),
    ("Soccer fans around the world celebrated the World Cup victory.", "sports"),
    ("The tennis player served an ace to win the match point.", "sports"),
    ("The Olympic athlete broke the world record by two seconds.", "sports"),
    ("The coach implemented a new defensive strategy for the playoffs.", "sports"),
    ("The baseball game went into extra innings with a tied score.", "sports"),
    ("The swimmer trained for months to qualify for the national team.", "sports"),
    ("The boxer landed a powerful right hook in the final round.", "sports"),
    ("The marathon runner crossed the finish line after four hours.", "sports"),
    ("The golf tournament attracted players from around the world.", "sports"),
    ("The hockey team scored two goals in the third period to win.", "sports"),
    ("The gymnast performed a flawless routine on the balance beam.", "sports"),
    ("The cyclist climbed the mountain stage and took the yellow jersey.", "sports"),
    ("The fans cheered loudly as their team scored the winning goal.", "sports"),
    ("The sprinter exploded off the blocks and never looked back.", "sports"),
    ("The volleyball team executed a perfect spike to win the set.", "sports"),
    ("The skier navigated the slalom course with incredible precision.", "sports"),
    ("The wrestler pinned his opponent in the first round.", "sports"),
    ("The Formula One driver took the lead on the final lap.", "sports"),
    # Technology
    ("The new smartphone features a revolutionary camera system.", "technology"),
    ("The software update fixed several security vulnerabilities.", "technology"),
    ("The startup raised 10 million dollars for their AI platform.", "technology"),
    ("The processor performance has doubled compared to last generation.", "technology"),
    ("The cloud service experienced a brief outage affecting users globally.", "technology"),
    ("The algorithm achieved state-of-the-art results on the benchmark.", "technology"),
    ("The company released a new version of its operating system.", "technology"),
    ("The wireless charging technology now supports faster speeds.", "technology"),
    ("The database query was optimized to run in under a millisecond.", "technology"),
    ("The encryption protocol was updated to address potential weaknesses.", "technology"),
    ("The machine learning model was trained on a dataset of millions of images.", "technology"),
    ("The programming language introduced a new type system feature.", "technology"),
    ("The electric vehicle battery achieved a range of 500 miles.", "technology"),
    ("The satellite internet service expanded to rural areas.", "technology"),
    ("The virtual reality headset offers an immersive gaming experience.", "technology"),
    ("The open source library received contributions from hundreds of developers.", "technology"),
    ("The network infrastructure was upgraded to support 10 gigabit speeds.", "technology"),
    ("The new GPU can render complex scenes in real time.", "technology"),
    ("The quantum computing breakthrough promises to revolutionize cryptography.", "technology"),
    ("The autonomous vehicle completed a cross-country trip without incidents.", "technology"),

    ("The pitcher struck out twelve batters in a dominant performance.", "sports"),
    ("The coach's strategy paid off with a last-minute victory.", "sports"),
    ("The swimmer broke the national record in the 100m freestyle.", "sports"),
    ("The boxer delivered a knockout punch in the final round.", "sports"),
    ("The marathon runner crossed the finish line with a personal best.", "sports"),
    ("The golf champion sank a thirty-foot putt to win the tournament.", "sports"),
    ("The hockey team scored three goals in the final period.", "sports"),
    ("The gymnast executed a flawless routine on the balance beam.", "sports"),
    ("The cyclist climbed the mountain stage with incredible endurance.", "sports"),
    ("The skier flew down the slope at over ninety miles per hour.", "sports"),
    ("The basketball player scored a career-high forty points.", "sports"),
    ("The referee's controversial call changed the outcome of the game.", "sports"),
    ("The underdog team advanced to the playoffs against all odds.", "sports"),
    ("The sprinter's start was explosive leaving competitors in the dust.", "sports"),
    ("The stadium erupted as the home team scored the winning goal.", "sports"),
    ("The surfer rode the biggest wave of the competition.", "sports"),
    ("The relay team passed the baton flawlessly for a gold medal.", "sports"),
    ("The weightlifter set a new personal record in the clean and jerk.", "sports"),
    ("The fencer's quick reflexes secured a narrow victory.", "sports"),
    ("The rowing team crossed the finish line with a photo finish.", "sports"),
    ("The new processor uses a three-nanometer architecture.", "technology"),
    ("Blockchain technology is transforming supply chain management.", "technology"),
    ("The startup developed a breakthrough in battery technology.", "technology"),
    ("Edge computing reduces latency for real-time applications.", "technology"),
    ("The Internet of Things connects billions of devices worldwide.", "technology"),
    ("5G networks enable ultra-fast data transfer speeds.", "technology"),
    ("The company launched a new cloud computing platform.", "technology"),
    ("Augmented reality is changing how we interact with digital content.", "technology"),
    ("The new programming language promises better memory safety.", "technology"),
    ("Researchers achieved quantum supremacy with a new processor.", "technology"),
    ("The software update introduced several security patches.", "technology"),
    ("Automated machine learning tools make AI more accessible.", "technology"),
    ("The new framework simplifies building microservices architectures.", "technology"),
    ("A major data breach exposed millions of user records.", "technology"),
    ("The robot navigated a complex environment without human input.", "technology"),
    ("Encryption protocols are essential for secure communications.", "technology"),
    ("The new smartphone features a revolutionary camera system.", "technology"),
    ("Containerization changed how applications are deployed.", "technology"),
    ("The latest SSD offers lightning-fast read and write speeds.", "technology"),
    ("Computer vision algorithms achieved human-level accuracy.", "technology"),
]

TOXICITY_TRAIN = [
    # Toxic
    ("You are an idiot and have no idea what you are talking about.", "toxic"),
    ("This is the worst garbage I have ever read, you should be ashamed.", "toxic"),
    ("Nobody cares about your stupid opinion, just shut up.", "toxic"),
    ("You are a complete failure and a waste of space.", "toxic"),
    ("Go away and never come back, you moron.", "toxic"),
    ("This post is absolutely terrible and so are you.", "toxic"),
    ("Your ignorance is staggering, learn something before speaking.", "toxic"),
    ("I hope you get banned for posting this nonsense.", "toxic"),
    ("What a pathetic and embarrassing attempt at a contribution.", "toxic"),
    ("You should not be allowed to post here, you ruin everything.", "toxic"),
    ("This is completely worthless information, delete it.", "toxic"),
    ("You are so dumb you do not even realize how wrong you are.", "toxic"),
    ("Your comment is offensive and should be removed immediately.", "toxic"),
    ("Nobody asked for your terrible opinion, keep it to yourself.", "toxic"),
    ("You are literally the worst person on this platform.", "toxic"),
    ("This is spam and you should be reported.", "toxic"),
    ("What a ridiculous and uninformed take on the subject.", "toxic"),
    ("You clearly have no clue what you are talking about.", "toxic"),
    ("Stop posting your garbage content everywhere.", "toxic"),
    ("Your contribution is useless and adds nothing to the discussion.", "toxic"),
    # Safe
    ("Thank you for sharing your perspective on this topic.", "safe"),
    ("I appreciate the thoughtful analysis you have provided.", "safe"),
    ("That is an interesting point, I had not considered that before.", "safe"),
    ("Great explanation, this really helped me understand the issue.", "safe"),
    ("I respectfully disagree, but I see where you are coming from.", "safe"),
    ("Thanks for taking the time to write this detailed response.", "safe"),
    ("Could you elaborate on that point? I would love to learn more.", "safe"),
    ("This is a very well-written article, thank you for sharing.", "safe"),
    ("I found this information quite useful for my research.", "safe"),
    ("Your analysis is thorough and well-supported by evidence.", "safe"),
    ("That is a fair criticism, I will take that into consideration.", "safe"),
    ("Great question, let me try to answer it based on what I know.", "safe"),
    ("I agree with most of what you said, especially about the first point.", "safe"),
    ("This community is lucky to have contributors like you.", "safe"),
    ("Thank you for the constructive feedback, I will work on improving.", "safe"),
    ("I learned something new from reading your post today.", "safe"),
    ("That is a creative solution to the problem, well done.", "safe"),
    ("Could you recommend any resources to learn more about this?", "safe"),
    ("Your work on this project is really impressive.", "safe"),
    ("I appreciate how you handled that difficult situation.", "safe"),

    ("You are completely useless do us all a favor and disappear.", "toxic"),
    ("Nobody wants to hear your stupid opinions shut up already.", "toxic"),
    ("You are a pathetic loser who will never accomplish anything.", "toxic"),
    ("This is the most idiotic thing I have ever read.", "toxic"),
    ("I hope you get what you deserve worthless human being.", "toxic"),
    ("You should be banned from posting forever.", "toxic"),
    ("Your ignorance is astounding go read a book sometime.", "toxic"),
    ("Go crawl back under whatever rock you came from.", "toxic"),
    ("You are a waste of space and a disgrace to society.", "toxic"),
    ("Nobody cares about your pathetic little opinion.", "toxic"),
    ("You are so dumb you could not pour water out of a boot.", "toxic"),
    ("This is garbage just like the person who wrote it.", "toxic"),
    ("Get a life you sad and pathetic excuse for a human.", "toxic"),
    ("Your presence here makes the whole platform worse.", "toxic"),
    ("Do the world a favor and log off permanently.", "toxic"),
    ("You are clearly too brain-dead to understand basic concepts.", "toxic"),
    ("I genuinely hope you fail at everything you try.", "toxic"),
    ("You are a disgusting piece of trash.", "toxic"),
    ("Go away nobody wants you here.", "toxic"),
    ("I want you to suffer for what you did.", "toxic"),
    ("Thank you for your thoughtful response it was very helpful.", "safe"),
    ("I appreciate you taking the time to explain that.", "safe"),
    ("That is a great point I had not considered that perspective.", "safe"),
    ("Could you please provide more details about your proposal?", "safe"),
    ("I completely agree with your analysis well said.", "safe"),
    ("Thank you for sharing your experience it is valuable insight.", "safe"),
    ("That is an interesting approach I would love to learn more.", "safe"),
    ("I see where you are coming from let me offer a different view.", "safe"),
    ("Your comment made me think about this issue differently.", "safe"),
    ("Great question Here is what I know about the topic.", "safe"),
    ("I really enjoyed reading your post thanks for sharing.", "safe"),
    ("That is very kind of you to offer your assistance.", "safe"),
    ("Could you clarify what you mean by that statement?", "safe"),
    ("I think there is merit to both sides of this argument.", "safe"),
    ("Wonderful job on the project your hard work shows.", "safe"),
    ("I am happy to help you with that problem.", "safe"),
    ("That makes perfect sense thank you for the explanation.", "safe"),
    ("Your dedication to this cause is truly inspiring.", "safe"),
    ("Let me know if there is anything I can do to support you.", "safe"),
    ("I really appreciate your honesty and transparency.", "safe"),
]



# ── Task definitions ─────────────────────────────────────────────────────


@dataclass
class QwenTask:
    """A simple text task for the Qwen CL experiment."""
    name: str
    domain: str
    data: List = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        labels = sorted(set(label for _, label in self.data))
        return len(labels)

    @property
    def labels(self) -> List[str]:
        return sorted(set(label for _, label in self.data))


TASK_A = QwenTask(name="Sentiment", domain="sentiment", data=SENTIMENT_TRAIN)
TASK_B = QwenTask(name="Topic", domain="topic", data=TOPIC_TRAIN)
# TASK_C = QwenTask(name="QuestionType", domain="question_type", data=QUESTION_TYPE_TRAIN)  # disabled
TASK_D = QwenTask(name="Toxicity", domain="toxicity", data=TOXICITY_TRAIN)

QWEN_CL_TASKS = [TASK_A, TASK_B, TASK_D]  # TASK_C (QuestionType) disabled


def make_prompt(text: str, domain: str) -> str:
    """Format a text example as an instruction-following prompt."""
    instructions = {
        "sentiment": "Classify the sentiment of the following movie review as 'positive' or 'negative'.",
        "topic": "Classify whether the following sentence is about 'sports' or 'technology'.",
        # "question_type": "disabled",
        "toxicity": "Classify the following text as 'toxic' or 'safe'.",
    }
    instr = instructions.get(domain, "Classify the following text.")
    # if domain == "question_type":
    #     return f"{instr}\n\nQuestion: {text}\n\nLabel:"
    return f"{instr}\n\nText: {text}\n\nLabel:"


def build_qwen_dataset(task: QwenTask, seed: int = 42, eval_split: float = 0.2,
                       tokenizer=None) -> tuple:
    """Build train/eval datasets for a Qwen task.

    If *tokenizer* is provided, examples are formatted with the model's
    chat template so the training/eval distribution matches what the
    instruct-tuned model expects.

    Returns (train_dataset, eval_dataset) as HuggingFace Datasets.
    Each example has:
      - 'text' (chat-formatted prompt + answer for causal LM training)
      - 'prompt' (chat-formatted prompt only, for evaluation generation)
      - 'target' (label only, for evaluation comparison)
    """
    rng = random.Random(seed)
    # Exact duplicate examples must never be split across train/eval.
    unique_data = list(dict.fromkeys(task.data))
    indexes = list(range(len(unique_data)))
    rng.shuffle(indexes)

    split_idx = int(len(indexes) * (1.0 - eval_split))
    train_idx = indexes[:split_idx]
    eval_idx = indexes[split_idx:]

    def _format_chat(text: str, label: str, domain: str, include_answer: bool) -> str:
        """Apply Qwen chat template to a single example."""
        instructions = {
            "sentiment": "Classify the sentiment of the following movie review as 'positive' or 'negative'.",
            "topic": "Classify whether the following sentence is about 'sports' or 'technology'.",
            # "question_type": "disabled",
            "toxicity": "Classify the following text as 'toxic' or 'safe'.",
        }
        system_msg = instructions.get(domain, "Classify the following text.")
        # if domain == "question_type":
        #     user_msg = f"Question: {text}"
        # else:
        user_msg = f"Text: {text}"  # simplified without question_type

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        if include_answer:
            messages.append({"role": "assistant", "content": label})

        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=not include_answer,
        )

    def _build(ids):
        texts = []
        prompts = []
        targets = []
        for i in ids:
            text, label = unique_data[i]
            if tokenizer is not None:
                full_text = _format_chat(text, label, task.domain, include_answer=True)
                prompt_text = _format_chat(text, label, task.domain, include_answer=False)
            else:
                prompt_text = make_prompt(text, task.domain)
                full_text = prompt_text + " " + label
            texts.append(full_text)
            prompts.append(prompt_text)
            targets.append(label)
        return Dataset.from_dict({"text": texts, "prompt": prompts, "target": targets})

    return _build(train_idx), _build(eval_idx)
