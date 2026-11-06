### NAVSECURA TECHNOLOGIES

## Computer Vision Engineering Assessment

Video understanding, tracking and event memory

OBJECTIVE

Conventional video analytics often reports the same person repeatedly across frames. The objective of this assessment is to build a small prototype that connects visual detections over time and produces useful, non-repetitive events.

SCENARIO

Use one CCTV-style video showing people moving through a scene. The video may come from an appropriately licensed public dataset or a self-recorded sample. Do not use private customer or employer footage.

IMPLEMENTATION

1. Use an appropriate open-source model to detect people in the video.
2. Track detected people across frames using temporary tracking identifiers.
3. Define a region of interest, such as an entrance, doorway or showroom area.
4. Record when a tracked person enters the region and how long they remain.
5. Produce one coherent event per visit rather than repeated frame-by-frame alerts.
6. Save an event history with timestamps, track identifier, duration and a reference to the relevant frame or short clip.

ENGINEERING DISCUSSION

Provide a concise explanation covering the following:

- Why you chose the detection and tracking approach.

- What happens when a person is temporarily occluded or leaves and re-enters the view.

- How you would extend the approach across cameras covering different areas of one site.

- How the system could learn recurring activity patterns without retraining a neural network every day.

DELIVERABLES

Share a repository containing the source code, setup instructions, and any model or data attribution; a short demonstration of the working prototype; and a brief description of a limitation or failure you encountered and what you would improve.

Existing open-source models and libraries may be used. Training a new model is not required. We are evaluating engineering judgement, clarity of reasoning and the working implementation.

NAVSECURA TECHNOLOGIES | ENGINEERING ASSESSMENT 0