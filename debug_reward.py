import os, string, copy as cp

def can_infer_option(answer, choices):
    verbose = os.environ.get('VERBOSE', 0)
    # Choices is a dictionary
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3 and verbose:
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False

def can_infer_text(answer, choices):
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False

def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)


def build_choices(choices):
    ret = {}
    for option, choice in zip(['A', 'B', 'C', 'D'], choices):
        if choice is not None:
            ret[option] = choice
    return ret


def srbench_accuracy_reward(completions, answer, A, B, C, D, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol, a, b, c, d in zip(contents, answer, A, B, C, D):

        choices = build_choices([a, b, c, d])
        ret = can_infer(content, choices)

        if ret == sol:
            reward = 1.0
        else:
            reward = 0.0

        rewards.append(reward)

    return rewards

if __name__ == "__main__":
    prompts=[[{'content': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>', 'role': 'system'}, {'content': '<image>\nQuestion: Consider the real-world 3D orientations of the objects. What is the relationship between the orientations of the bicycle and the motorcycle, parallel of perpendicular to each other?\nOptions:\nA. parallel\nB. perpendicular\nPlease select the correct answer from the options above.', 'role': 'user'}], [{'content': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>', 'role': 'system'}, {'content': '<image>\nQuestion: Consider the real-world 3D orientations of the objects. What is the relationship between the orientations of the bicycle and the motorcycle, parallel of perpendicular to each other?\nOptions:\nA. parallel\nB. perpendicular\nPlease select the correct answer from the options above.', 'role': 'user'}], [{'content': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>', 'role': 'system'}, {'content': '<image>\nQuestion: Consider the real-world 3D orientations of the objects. What is the relationship between the orientations of the bicycle and the motorcycle, parallel of perpendicular to each other?\nOptions:\nA. parallel\nB. perpendicular\nPlease select the correct answer from the options above.', 'role': 'user'}], [{'content': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>', 'role': 'system'}, {'content': '<image>\nQuestion: Consider the real-world 3D orientations of the objects. What is the relationship between the orientations of the bicycle and the motorcycle, parallel of perpendicular to each other?\nOptions:\nA. parallel\nB. perpendicular\nPlease select the correct answer from the options above.', 'role': 'user'}], [{'content': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>', 'role': 'system'}, {'content': '<image>\nQuestion: Consider the real-world 3D orientations of the objects. What is the relationship between the orientations of the bicycle and the motorcycle, parallel of perpendicular to each other?\nOptions:\nA. parallel\nB. perpendicular\nPlease select the correct answer from the options above.', 'role': 'user'}], [{'content': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>', 'role': 'system'}, {'content': '<image>\nQuestion: Consider the real-world 3D orientations of the objects. What is the relationship between the orientations of the bicycle and the motorcycle, parallel of perpendicular to each other?\nOptions:\nA. parallel\nB. perpendicular\nPlease select the correct answer from the options above.', 'role': 'user'}], [{'content': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>', 'role': 'system'}, {'content': '<image>\nQuestion: Consider the real-world 3D orientations of the objects. What is the relationship between the orientations of the bicycle and the motorcycle, parallel of perpendicular to each other?\nOptions:\nA. parallel\nB. perpendicular\nPlease select the correct answer from the options above.', 'role': 'user'}], [{'content': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>', 'role': 'system'}, {'content': '<image>\nQuestion: Consider the real-world 3D orientations of the objects. What is the relationship between the orientations of the bicycle and the motorcycle, parallel of perpendicular to each other?\nOptions:\nA. parallel\nB. perpendicular\nPlease select the correct answer from the options above.', 'role': 'user'}]] 
    completions=[[{'role': 'assistant', 'content': '\n\nB'}], [{'role': 'assistant', 'content': '\n\nA'}], [{'role': 'assistant', 'content': '\n\nB'}], [{'role': 'assistant', 'content': '\n\nA'}], [{'role': 'assistant', 'content': '\n\nSince'}], [{'role': 'assistant', 'content': '\n\nC'}], [{'role': 'assistant', 'content': '\n\nA'}], [{'role': 'assistant', 'content': '\n\nA'}]]
    reward_kwargs={'index': ['Y6LOQHT1', 'Y6LOQHT1', 'Y6LOQHT1', 'Y6LOQHT1', 'Y6LOQHT1', 'Y6LOQHT1', 'Y6LOQHT1', 'Y6LOQHT1'], 'question': ['Consider the real-world 3D locations of the objects. Is the airplane directly above the chair?', 'Consider the real-world 3D locations of the objects. Is the airplane directly above the chair?', 'Consider the real-world 3D locations of the objects. Is the airplane directly above the chair?', 'Consider the real-world 3D locations of the objects. Is the airplane directly above the chair?', 'Consider the real-world 3D locations of the objects. Is the airplane directly above the chair?', 'Consider the real-world 3D locations of the objects. Is the airplane directly above the chair?', 'Consider the real-world 3D locations of the objects. Is the airplane directly above the chair?', 'Consider the real-world 3D locations of the objects. Is the airplane directly above the chair?'], 'A': ['yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes'], 'B': ['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no'], 'C': [None, None, None, None, None, None, None, None], 'D': [None, None, None, None, None, None, None, None], 'answer': ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'], 'category': ['location_above', 'location_above', 'location_above', 'location_above', 'location_above', 'location_above', 'location_above', 'location_above'], 'image_source': ['MS-COCO', 'MS-COCO', 'MS-COCO', 'MS-COCO', 'MS-COCO', 'MS-COCO', 'MS-COCO', 'MS-COCO'], 'image_url': ['http://images.cocodataset.org/train2017/000000233079.jpg', 'http://images.cocodataset.org/train2017/000000233079.jpg', 'http://images.cocodataset.org/train2017/000000233079.jpg', 'http://images.cocodataset.org/train2017/000000233079.jpg', 'http://images.cocodataset.org/train2017/000000233079.jpg', 'http://images.cocodataset.org/train2017/000000233079.jpg', 'http://images.cocodataset.org/train2017/000000233079.jpg', 'http://images.cocodataset.org/train2017/000000233079.jpg']}
    output_reward_func = srbench_accuracy_reward(completions=completions, **reward_kwargs)