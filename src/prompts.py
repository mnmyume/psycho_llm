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

EXPERT_REASONING_PROMPT = """Task:
You are an expert spatial and environmental analyzer evaluating isometric sandbox images. Rate the image on two dimensions on a scale of 1 to 3.

CRITICAL INSTRUCTION: Do not judge "Tidy" or "Moderate" based on whether the image looks like a clean pixel-art asset or an aligned grid. Base your score STRICTLY on the physical, touching connectivity of the path.

Dimension 1: Chaos (1) to Tidy (3)
Evaluate the structural integrity, flow, and physical connection of the brown paths.

Score 1 (Chaotic): The path is severely shattered or fragmented. KNOCK-OUT RULE: If the path consists of multiple completely isolated brown squares (islands), scattered blocks, or is broken into many disconnected segments that do not physically touch, it MUST be a 1. Do NOT upgrade to a 2 just because the disconnected blocks align to an isometric grid or look like a "map". If it is a collection of islands, it is a 1.

Score 2 (Moderate): The path is mostly continuous and traceable as a single main line, but suffers from a specific structural flaw. It must physically connect for most of the image, but might suffer from a noticeable gap, an interruption (like steps breaking the dirt line), or stop abruptly at a single dead end instead of finishing its route.

Score 3 (Tidy): The layout is highly intentional and structured. The path is 100% continuous and flawless. It must travel entirely from edge-to-edge or form a complete connected loop with ZERO dead ends and ZERO breaks.

Dimension 2: Monotony (1) to Variety (3)
Evaluate the diversity of the environmental elements.

Score 1 (Monotonous): The environment is barren. It consists of 0 distinct assets (only flat snow and a simple brown path) with no flora or terrain changes.

Score 2 (Moderate): The environment introduces exactly 1 type of asset, such as a single uniform species of tree, or minor terrain variations (raised dirt blocks/steps), but remains visually straightforward.

Score 3 (Variety): The environment is rich. Look for 2 or more distinct elements coexisting: different types of vegetation (e.g., combining tall pines and frosted trees), water features, or prominent terrain elevation.

Evaluation Process:
First, trace the brown paths. Is it a collection of shattered islands (1), a mostly connected line with a flaw/dead-end (2), or perfectly continuous (3)? Ignore implied grids.
Second, inventory the elements. How many different types of assets (trees, water, elevation) are present?

Output Format:
Output strictly raw JSON. Do not use markdown blocks. Use the reasoning key to briefly outline your step-by-step observations before outputting the scores.
{
"reasoning": "A 1-2 sentence explanation of the path continuity (specifically noting dead ends/breaks) and element diversity.",
"chaos_tidy_score": [Insert integer 1-3],
"monotony_variety_score": [Insert integer 1-3]
}"""

# A dictionary to easily look up prompts by name
PROMPTS = {
    "basic": BASIC_PROMPT,
    "expert_reasoning": EXPERT_REASONING_PROMPT,
}
