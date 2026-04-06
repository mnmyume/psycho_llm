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

CRITICAL INSTRUCTION: Do not judge "Tidy" or "Moderate" based on whether the image looks like a clean pixel-art asset or an aligned grid. Base your score STRICTLY on the overall macro-connectivity of the brown path.

Dimension 1: Chaos (1) to Tidy (3)
Evaluate the structural integrity, flow, and physical connection of the brown paths.

Important Caveat on Texture: Treat the brown path as a solid object. Do NOT interpret the internal cobblestone/dirt pixel texture as "gaps" or "breaks."

Score 1 (Chaotic): The path is severely shattered or fragmented at a macro level. KNOCK-OUT RULE: If the path consists of multiple completely isolated brown squares (islands), scattered blocks, or is broken into many disconnected segments separated by snow that do not physically touch, it MUST be a 1. Do NOT upgrade to a 2 just because the disconnected blocks align to an isometric grid. If it is a collection of islands, it is a 1.

Score 2 (Moderate): The path is mostly continuous and traceable as a single main line, but suffers from a specific macro-structural flaw. It must physically connect for most of the image, but suffers from a noticeable gap, an interruption (like steps breaking the dirt line), or stops abruptly at a single dead end in the middle of the snow instead of finishing its route.

Score 3 (Tidy): The layout is highly intentional and structured. The path is 100% continuous and flawless. It forms a complete connected loop or travels entirely across the terrain block (exiting the visible terrain boundaries) with ZERO dead ends, ZERO breaks, and ZERO isolated islands. Clear, continuous crossroads (like an X or + shape) are Score 3.

Dimension 2: Monotony (1) to Variety (3)
Evaluate the diversity of the environmental elements.

Score 1 (Monotonous): The environment is barren. It consists of 0 distinct assets (only flat snow and a simple brown path) with no flora or terrain changes.

Score 2 (Moderate): The environment introduces exactly 1 type of asset, such as a single uniform species of tree, or minor terrain variations (raised dirt blocks/steps), but remains visually straightforward.

Score 3 (Variety): The environment is rich. Look for 2 or more distinct elements coexisting: different types of vegetation (e.g., combining tall pines and frosted trees), water features, or prominent terrain elevation (large snow mounds or raised platforms).

Evaluation Process:
First, trace the macroscopic brown path. Is it a collection of shattered islands separated by snow (1), a mostly connected line with a flaw/dead-end (2), or perfectly continuous with zero dead ends (3)? Ignore the internal pixel texture of the dirt.
Second, inventory the elements. How many different types of assets (trees, water, elevation) are present?

Output Format:
Output strictly raw JSON. Do not use markdown blocks. Use the reasoning key to briefly outline your step-by-step observations before outputting the scores.
{
"reasoning": "A 1-2 sentence explanation of the path continuity (specifically noting dead ends/breaks) and element diversity.",
"chaos_tidy_score": [Insert integer 1-3],
"monotony_variety_score": [Insert integer 1-3]
}"""

GRID_TEST_PROMPT="""
You are given an 8×8 rhombus-shaped grid (isometric view), where each cell is a diamond (rhombus). Exactly one cell is colored blue.

Return the location of the blue cell as grid indices [x, y], not pixel coordinates.

Coordinate system definition:

The topmost cell is [0, 0].

All coordinates satisfy: 0 ≤ x ≤ 7 and 0 ≤ y ≤ 7.

Strict instructions:

You must count from [0,0] explicitly.
Trace the diagonal (right-edge direction) and count x step-by-step.
Trace the diagonal (left-edge direction) and count y step-by-step.
Do not estimate visually.
Reason briefly and keep the counting short before answering.

Output format:
Return strictly raw JSON (no markdown, no explanation):
{
"coordinates": [x, y]
}
"""

GRID_003_SYSTEM_PROMPT = (
    "You are an expert spatial reasoning AI specializing in isometric game engines. "
    "Your task is to analyze an isometric scene and map the visual application of a specific texture to its corresponding grid index.\n\n"
    "Multimodal Input Structure:\n"
    "You will be provided with two images and one set of numerical metadata for each analysis:\n"
    "1.  **Image A (Isometric Sandbox):** The primary scene view showing an 8×8 isometric grid. "
    "The selected cell is indicated by a red-colored texture applied to it.\n"
    "2.  **Image B (Brush Texture):** A 64x64 square texture image (the \"brush\"). In the sandbox image, this square texture is projected onto a specific grid cell, transforming it visually into an isometric diamond shape.\n"
    "3.  **Metadata (Pixel Boundary):** A numerical array representing the standard 2D pixel bounding box `[originalX, originalY, width, height]` of the applied texture brush *within Image A (Isometric Sandbox)*.\n\n"
    "Mapping Rules:\n"
    "1.  The grid is exactly 8×8. Valid indices range from 0 to 7 for both `i` and `j`. Never output a value outside [0, 7].\n"
    "2.  The topmost cell of the diamond is `[0, 0]`.\n"
    "3.  `i` axis: increases going DOWN-RIGHT from the top. The cells along the top-right edge of the diamond are `[0,0], [1,0], [2,0], ..., [7,0]`.\n"
    "4.  `j` axis: increases going DOWN-LEFT from the top. The cells along the top-left edge of the diamond are `[0,0], [0,1], [0,2], ..., [0,7]`.\n"
    "5.  The bottom-most cell of the diamond is `[7, 7]`.\n\n"
    "Before answering, verify:\n"
    "- Are both `i` and `j` in the range [0, 7]?\n"
    "- Is `i` the position along the down-RIGHT diagonal, and `j` along the down-LEFT diagonal? Do not swap them.\n\n"
    "Output Mandate:\n"
    "Based on the provided images and pixel boundary, calculate the precise grid index. "
    'You must return ONLY valid JSON format with the coordinates: `{"index": [i, j]}`. Do not include any conversational text.'
)

GRID_003_USER_PROMPT_TEMPLATE = (
    "Analyzing current scene and brush. "
    "<image_0> (Isometric Sandbox) <image_1> (64x64 Brush Texture)"
    "Pixel boundary: [{pixel_boundary}]"
)

GRID_004_SYSTEM_PROMPT = (
    "You are an expert spatial reasoning AI. Your task is to identify the grid "
    "coordinates of a highlighted cell in an 8×8 isometric grid image.\n\n"
    "The image shows an isometric (diamond-shaped) 8×8 grid. Each cell's index "
    "is labeled with numbers along the grid edges, ranging from 0 to 7 on both "
    "axes. Exactly one cell is colored red.\n\n"
    "Coordinate System:\n"
    "- The topmost cell is [0, 0].\n"
    "- The first axis (i) increases going DOWN-RIGHT from the top.\n"
    "- The second axis (j) increases going DOWN-LEFT from the top.\n"
    "- All valid indices are in [0, 7].\n\n"
    "Output Mandate:\n"
    "Return ONLY valid JSON with the coordinates of the red cell: "
    '{"coordinates": [i, j]}. Do not include any other text.'
)

GRID_004_USER_PROMPT = (
    "The numbers in this image indicate the index of each grid cell along its "
    "respective axis. Identify the grid coordinates of the red cell."
)

# A dictionary to easily look up prompts by name
PROMPTS = {
    "basic": BASIC_PROMPT,
    "expert_reasoning": EXPERT_REASONING_PROMPT,
    "grid_test": GRID_TEST_PROMPT,
    "grid_003": GRID_003_USER_PROMPT_TEMPLATE,
    "grid_004": GRID_004_USER_PROMPT,
}
