"""
Provides various prompts for the psycho-llm dataset generation, allowing you to iterate
on different instructions for the vision model while keeping historical prompts safe.
"""

BASIC_PROMPT = (
    "Analyze this isometric sandbox image. "
    "Rate it on two dimensions from 1 to 5. "
    "Dimension 1: Chaos to Tidy (1=chaotic/fragmented, 5=tidy/smooth). "
    "Dimension 2: Monotony to Variety (1=monotonous/simple, 5=variety/complicated). "
    "Output strict JSON with keys 'chaos_tidy_score' and 'monotony_variety_score'."
)

EXPERT_REASONING_PROMPT = """You are an expert spatial and environmental analyzer evaluating isometric sandbox images. Your task is to analyze the psychological landscape presented in the image and rate it on two specific dimensions on a scale of 1 to 3.

Dimension 1: Chaos (1) to Tidy (3)
Evaluate the structural integrity, flow, and organization of the layout.

Score 1 (Chaotic): The layout is highly disorganized. Look for highly fragmented, disconnected brown paths, dead ends, or scattered, noisy placements of elements. The terrain lacks a cohesive flow.

Score 2 (Moderate): The layout has a general direction but contains noticeable interruptions. Paths might be mostly continuous but weave wildly, or the surrounding elements (like dense forests) create an unstructured, organic feel.

Score 3 (Tidy): The layout is highly intentional, structured, and smooth. Look for continuous, unbroken paths, clear crossroads, symmetrical placements (e.g., balanced snow mounds or organized tree lines), and smooth integrations of rivers and bridges.

Dimension 2: Monotony (1) to Variety (3)
Evaluate the diversity and richness of the environmental elements.

Score 1 (Monotonous): The environment is barren, basic, or highly repetitive. It consists of only one or two element types (e.g., only flat snow and a simple brown path) with no flora or terrain changes.

Score 2 (Moderate): The environment introduces a few new elements, such as a single type of tree or minor terrain variations, but remains visually straightforward.

Score 3 (Variety): The environment is rich and diverse. Look for multiple distinct elements coexisting: different types of vegetation (e.g., pine trees and cherry blossoms), water features (rivers, log bridges), terrain elevation (snow mounds, raised blocks), and varied ground textures (grass patches).

Evaluation Process:

First, trace the brown paths and rivers. Are they fragmented or continuous?

Second, inventory the elements. How many different types of assets (trees, water, elevation) are present?

Synthesize these observations to assign your final scores.

Output Format:
Output strictly raw JSON. Do not use markdown blocks. Your JSON must contain exactly three keys. Use the reasoning key to briefly outline your step-by-step observations before outputting the scores.
{
"chaos_tidy_score": [Insert integer 1-3],
"monotony_variety_score": [Insert integer 1-3]
}"""

# A dictionary to easily look up prompts by name
PROMPTS = {
    "basic": BASIC_PROMPT,
    "expert_reasoning": EXPERT_REASONING_PROMPT,
}
