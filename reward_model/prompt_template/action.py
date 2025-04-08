ACTION_PROMPT = """
You are a video description expert. You will receive an AI-generated video.
You need to watch the video carefully and then recognize the actions performed by the subject in the video. 
Please provide a rating from 1 to 3 for the actions performed by the subject in the video according to the "Scoring Strategy", with 1 being the worst and 3 being the best.

### Important Notes:
1. Determine if the action of the subject is recognizable.
2. Consider the clarity of the action performed, including whether it is fully completed or distorted in any way.

### Scoring Strategy:
Your scoring should focus on the action(s) performed by the subject in the video. Please follow these steps:

1. Action Accuracy: Whether the video shows the exact action in the text prompt without deviation into an unrelated or similar action.
2. Action Consistency: Whether the video maintains the same action throughout the video, without changing into another action that doesn‘t align with the text prompt.
3. Action Completion: Does the subject in the video complete all the actions in the text prompt without missing any actions.
4. You only need to focus on action consistency and do not need to consider factors such as the number of subjects or the scene.

### Scoring Range
Then based on the above considerations, you need to assign a specific score from 1 to 3 for each video(from 1 to 3, with 3 being the highest quality,using increments of 1) according to the 'Scoring Range':

1. Poor consistency - The subject's action does not match the text prompt at all, or the subject's action is not recognized, or the subject in the text prompt does not appear.
2. Moderate consistency - The main subject’s action is barely recognizable but imperfectly generated, specifically meeting one or more of the following conditions:
    - Condition 1 : The main subject's action is incomplete and does not fully perform the action in the text prompt.
    - Condition 2 : The main subject’s action has significant deviations in appearance or process compared to the real action, making it distorted and hard to recognize.
    - Condition 3 : The subject's action changes into another action that is inconsistent with the action in the text prompt.
    - Condition 4 : A similar action is generated, such as marching instead of parading.
    - Condition 5 : The subject in the video did not complete all of the actions in the text prompt, but only partially completed them
3.  Good consistency  - The action fully aligns with the text prompt, is accurate, complete, and clearly recognizable, without any abrupt changes in the action.

Textual prompt - {text_prompt}
Please provide the score of actions: <|reward|>
END
"""

