from fpdf import FPDF

# Create a class for the PDF
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Project Outline: Spectral Analysis of Toy Models', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(230, 230, 230)  # Light gray background
        self.cell(0, 8, title, 0, 1, 'L', True)
        self.ln(2)

    def section_body(self, text):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 5, text)
        self.ln()

    def bullet_point(self, text):
        self.set_font('Arial', '', 11)
        self.cell(5)  # Indent
        self.multi_cell(0, 5, f"{chr(149)} {text}")
        self.ln(1)

# Initialize PDF
pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# --- PHASE 1 ---
pdf.section_title("Phase 1: Theoretical Framework & Definitions")
pdf.section_body("Objective: Establish the mathematical ground rules based on the 'Mathematical Framework for Transformers' (Elhage et al., 2021).")

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, '1. Define the QK Circuit (The "Where"):', 0, 1)
pdf.bullet_point("Formula: W_QK_Full = W_E_transpose * W_QK_head * W_E")
pdf.bullet_point("Goal: Understand that this circuit provides attention scores and determines WHERE information moves (the attention pattern).")
pdf.ln(2)

pdf.set_font('Arial', 'B', 11)
pdf.cell(0, 6, '2. Define the OV Circuit (The "What"):', 0, 1)
pdf.bullet_point("Formula: W_OV_Full = W_U * W_OV_head * W_E")
pdf.bullet_point("Goal: Understand that this circuit determines WHAT information is moved and how it affects final logits (e.g., copying or preserving information).")
pdf.ln(4)

# --- PHASE 2 ---
pdf.section_title("Phase 2: Model Selection & Data Extraction")
pdf.section_body("Objective: Select specific 'toy' models to compare deep reasoning capabilities against standard behavior.")
pdf.bullet_point("Select Model: TinyStories-33M (a reasoning-capable toy model).")
pdf.bullet_point("Control Model: Use a randomly initialized version or a 1-layer version of the same model.")
pdf.bullet_point("Extract Weights: Access Embedding (W_E), Unembedding (W_U), and Head weights (W_Q, W_K, W_V, W_O).")
pdf.ln(4)

# --- PHASE 3 ---
pdf.section_title("Phase 3: Spectral Computation")
pdf.section_body("Objective: Perform the core mathematical operations to generate data.")
pdf.bullet_point("Compute Full Matrices: Calculate the vocab-by-vocab matrices for both QK and OV circuits.")
pdf.bullet_point("Eigen-Decomposition: Compute eigenvalues and eigenvectors.")
pdf.bullet_point("Focus Area: Look for a tail-end distribution of eigenvalues at 0.")
pdf.bullet_point("Density Estimation: Plot the distribution of these eigenvalues.")
pdf.ln(4)

# --- PHASE 4 ---
pdf.section_title("Phase 4: Comparative Analysis (The Core Experiment)")
pdf.section_body("Objective: Answer specific research questions regarding reasoning and evolution.")
pdf.bullet_point("Compare Signatures: Do reasoning models show a different spectral density than non-reasoning ones?")
pdf.bullet_point("Identify 'Thinking' Heads: Which specific heads evolved the most unique signatures?")
pdf.bullet_point("Interpret Eigenvalues: Do positive/negative values correspond to semantic scaling (e.g., Bad -> Terrible vs. Bad -> Good)?")
pdf.bullet_point("Rank Analysis: Test the null hypothesis that these matrices are high-rank.")
pdf.ln(4)

# --- PHASE 5 ---
pdf.section_title("Phase 5: Bias Detection & Conclusion")
pdf.section_body("Objective: Apply spectral analysis to detect specific model behaviors.")
pdf.bullet_point("Bias Probing: Use eigenvectors to detect biases (e.g., gender correlating with career terms).")
pdf.bullet_point("Conclusion: Validate if eigen-analysis is a viable tool for interpreting 'where thinking is happening' in LLMs.")

# Save
output_filename = "Project_Outline_Spectral_Analysis.pdf"
pdf.output(output_filename)
print(f"PDF generated: {output_filename}")