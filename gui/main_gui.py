# ===================== IMPORTS & PATH SETUP =====================
# Standard libraries for system operations and threading
import sys, os, threading

# Add project root directory to Python path so internal modules can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# (Repeated path setup – kept as-is to avoid breaking existing behavior)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Regular expressions for text pattern matching
import re

# Tkinter for GUI components
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# PIL for image loading and thumbnail creation
from PIL import Image, ImageTk

# Core ML prediction function
from scripts.predict_review import analyze_review

# OCR function to extract text from images
from scripts.ocr_reader import extract_text_from_image

# Text preprocessing utilities
from scripts.preprocess import (
    clean_ui_junk,     # removes UI-related noise
    split_reviews,     # splits OCR text into individual reviews
    is_valid_review,   # filters invalid/non-review text
    clean_text         # prepares text for ML model
)

# Matplotlib-based charts for visualization
from gui import charts

# Hashing for deduplication of reviews
import hashlib

# Logger utility for debugging and monitoring
from utils.logger import get_logger

# Initialize GUI logger
logger = get_logger("gui")
logger.info("GUI launched")


# ===================== LOADER / PROGRESS OVERLAY =====================
# Displays a semi-transparent overlay while OCR and ML processing runs

class LoaderOverlay:
    def __init__(self, parent):
        self.parent = parent
        self.top = None
        self.canvas = None
        self.bar = None
        self.animating = False
        self.progress_width = 0
       
    def show(self, text="Extracting & analyzing..."):
        # Prevent multiple loader overlays
        if self.top:
            return

        self.animating = True

        # Create overlay window above main GUI
        self.top = tk.Toplevel(self.parent)
        self.top.overrideredirect(True)
        self.top.configure(bg="#8DC3CA")
        self.top.attributes("-alpha", 0.55)

        # Match overlay size and position with main window
        self.top.geometry(
            f"{self.parent.winfo_width()}x{self.parent.winfo_height()}+"
            f"{self.parent.winfo_rootx()}+{self.parent.winfo_rooty()}"
        )

        # Canvas used to draw loader elements
        self.canvas = tk.Canvas(
            self.top,
            bg="#8DC3CA",
            highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        # Loader status text
        self.canvas.create_text(
            self.parent.winfo_width() // 2,
            self.parent.winfo_height() // 2 - 35,
            text=text,
            fill="#1E2A2B",
            font=("Segoe UI SemiBold", 16)
        )

        # Progress bar dimensions and placement
        bar_width = 240
        bar_height = 10
        x = (self.parent.winfo_width() - bar_width) // 2
        y = self.parent.winfo_height() // 2

        # Progress bar background
        self.bar_bg = self.canvas.create_rectangle(
            x, y, x + bar_width, y + bar_height,
            outline="#5F949E", fill="#E9F1F2"
        )

        # Progress bar foreground (animated)
        self.bar = self.canvas.create_rectangle(
            x, y, x, y + bar_height,
            fill="#124A49", outline=""
        )

        # Start animation loop
        self._animate()

    def _animate(self):
        # Stop animation when loader is hidden
        if not self.animating:
            return

        bar_width = 240
        step = 8  # animation speed

        # Move bar forward
        self.progress_width += step
        if self.progress_width > bar_width:
            self.progress_width = 0

        x = (self.parent.winfo_width() - bar_width) // 2
        y = self.parent.winfo_height() // 2

        # Update bar position
        self.canvas.coords(
            self.bar,
            x,
            y,
            x + self.progress_width,
            y + 10
        )

        # Schedule next animation frame
        self.top.after(20, self._animate)

    def hide(self):
        # Stop animation and remove overlay
        self.animating = False
        if self.top:
            self.top.destroy()
        self.top = None


# ===================== GENERIC KEYWORDS =====================
# Common promotional words frequently found in fake reviews

GENERIC_KEYWORDS = [
    "awesome", "excellent", "amazing", "superb", "very good",
    "great", "perfect", "best", "nice", "premium", "fantastic"
]

# Highlights generic phrases and keywords inside review text
def highlight_generic(text_widget, text, phrases):
    text_lower = text.lower()

    # 1️⃣ Highlight fuzzy-matched generic phrases detected by model
    for phrase in phrases:
        words = phrase.lower().split()
        pattern = r".{0,10}".join(map(re.escape, words))
        for m in re.finditer(pattern, text_lower):
            start = f"1.0+{m.start()}c"
            end = f"1.0+{m.end()}c"
            text_widget.tag_add("generic", start, end)

    # 2️⃣ Highlight known generic keywords
    for word in GENERIC_KEYWORDS:
        for m in re.finditer(rf"\b{word}\b", text_lower):   
            start = f"1.0+{m.start()}c"
            end = f"1.0+{m.end()}c"
            text_widget.tag_add("generic", start, end)


# ===================== MAIN APPLICATION CLASS =====================

class App:
    def __init__(self, root):
        self.root = root

        # Window title and size
        root.title("Product Review Analysis System")
        root.geometry("1150x780")
        root.configure(bg="#8DC3CA")

        # Loader overlay instance
        self.loader = LoaderOverlay(root)

        # ---------------- STYLE CONFIGURATION ----------------
        style = ttk.Style()

        style.configure("TFrame", background="#E9F1F2")
        style.configure("TLabel", background="#E9F1F2", foreground="#75C4C3")
        style.configure("Card.TFrame", background="#E9F1F2")
        style.configure(
            "CardTitle.TLabel",
            background="#FFFFFF",
            foreground="#1E2A2B",
            font=("Segoe UI", 11, "bold")
        )
        style.configure(
            "DarkLabel.TLabel",
            foreground="#1E2A2B",   # dark readable text
            background="#E9F1F2",   # panel background
            font=("Segoe UI", 10)
        )
        style.configure(
            "CardText.TLabel",
            background="#FFFFFF",
            foreground="#1E2A2B",
            wraplength=460,
            justify="left"
        )
        style.configure(
            "Grey.Horizontal.TScrollbar",
            troughcolor="#5F949E",
            background="#AFD5DE",
            bordercolor="#1E2A2B",
            arrowcolor="#1E2A2B",
            relief="flat"
        )
        style.configure(
            "Panel.TFrame",
            background="#E9F1F2",
        )

        # Application heading label
        ttk.Label(
            root,
            text="Product Review Analysis System",
            font=("Segoe UI", 18, "bold"),
            background="#8DC3CA",
            foreground="#3C6061"  
        ).pack(pady=10)

        # Tab container
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        # Tabs for manual input and OCR-based analysis
        self.tab_manual = ttk.Frame(notebook)
        self.tab_ocr = ttk.Frame(notebook)

        notebook.add(self.tab_manual, text="Manual Review")
        notebook.add(self.tab_ocr, text="Scan & Analyze Reviews")

        # Application state variables
        self.items = []
        self.selected_index = None
        self.results_container = None
        self.platform = tk.StringVar(value="Amazon")
        self.platform_var = tk.StringVar(value="Amazon")

        # Initialize both tabs
        self.setup_manual_tab()
        self.setup_ocr_tab()

    # ===================== MANUAL REVIEW TAB =====================
    def setup_manual_tab(self):         
        frame = ttk.Frame(self.tab_manual, padding=16, style="TFrame")
        frame.configure(style="TFrame")
        frame.configure(style="Panel.TFrame")
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Enter Review:", style="DarkLabel.TLabel").pack(anchor="w")

        # Text box where user enters a review manually
        self.manual_text = tk.Text(
            frame,
            height=6,
            bg="#FFFFFF",
            fg="#1E2A2B",
            insertbackground="#1E2A2B"
        )
        self.manual_text.pack(fill="x", pady=6)

        # Container for Analyze and Clear buttons
        btns = ttk.Frame(frame)
        btns.pack(pady=8)

        # Button to start manual review analysis
        ttk.Button(
            btns,
            text="Analyze",
            command=self.start_manual
        ).pack(side="left", padx=6)

        # Button to clear text and results
        ttk.Button(
            btns,
            text="Clear",
            command=self.clear_manual
        ).pack(side="left", padx=6)

        # Label to display prediction result and confidence
        self.manual_result = ttk.Label(frame, font=("Segoe UI", 11))
        self.manual_result.pack(pady=10)

    # Clears manual review input and output
    def clear_manual(self):
        self.manual_text.delete("1.0", "end")
        self.manual_result.config(text="")

    # Starts manual review analysis in a background thread
    def start_manual(self):
        text = self.manual_text.get("1.0", "end").strip()
    
        # Basic validation to avoid meaningless input
        if len(text.split()) < 3:
            messagebox.showwarning(
                "Invalid Review",
                "Please enter a meaningful product review."
            )
            return
    
        # Run analysis without freezing GUI
        threading.Thread(
            target=self._manual_worker,
            args=(text,),
            daemon=True
        ).start()

    # Worker function that runs ML model for manual review
    def _manual_worker(self, text):
        # Show loader overlay
        self.root.after(
            0,
            lambda: self.loader.show("Analyzing review...")
        )

        # Run prediction model
        pred, conf, meta = analyze_review(text)

        # Update UI safely from main thread
        self.root.after(
            0,
            lambda: (
                self.loader.hide(),
                self.manual_result.config(
                    text=f"Prediction: {pred}\nConfidence: {conf:.2f}%"
                )
            )
        )

    # ===================== OCR / IMAGE ANALYSIS TAB =====================
    def setup_ocr_tab(self):
        # Main container for OCR tab
        frame = ttk.Frame(self.tab_ocr, padding=10)
        frame.pack(fill="both", expand=True)

        # Top control bar (platform, buttons, charts)
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x")

        # Platform selection label
        ttk.Label(
            ctrl,
            text="Platform:",
            style="DarkLabel.TLabel"
        ).pack(side="left", padx=(0, 4))

        # Dropdown to select e-commerce platform
        platform_menu = ttk.Combobox(
            ctrl,
            textvariable=self.platform_var,
            values=[
                "Amazon",
                "Flipkart",
                "Myntra",
                "Nykaa",
                "Meesho"
            ],
            state="readonly",
            width=12,
            style="TCombobox"
        )
        platform_menu.pack(side="left", padx=6)

        # Button to add review screenshots
        ttk.Button(
            ctrl,
            text="Add Images",
            command=self.add_images
        ).pack(side="left", padx=6)

        # Button to analyze all added images
        ttk.Button(
            ctrl,
            text="Analyze All",
            command=self.start_analyze
        ).pack(side="left", padx=6)

        # Button to clear all loaded images and results
        ttk.Button(
            ctrl,
            text="Clear",
            command=self.clear_images
        ).pack(side="left", padx=6)

        # Chart selection variable
        self.chart_type = tk.StringVar(value="Summary")

        # Dropdown for selecting chart type
        chart_menu = ttk.Combobox(
            ctrl,
            textvariable=self.chart_type,
            values=[
                "Summary",
                "Average Confidence"
            ],
            state="readonly",
            width=20,
            style="TCombobox"
        )
        chart_menu.pack(side="right", padx=6)

        # Button to display selected chart
        ttk.Button(
            ctrl,
            text="Show Chart",
            command=self.show_charts
        ).pack(side="right", padx=6)

        # Main content area (left: images, right: results)
        content = ttk.Frame(frame)
        content.pack(fill="both", expand=True, pady=10)

        # Left panel: image thumbnails and preview
        left = ttk.Frame(content)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        # Right panel: OCR text and analysis cards
        right = ttk.Frame(content)
        right.pack(side="right", fill="both", expand=True)

        # Frame containing scrollable thumbnail canvas
        canvas_frame = ttk.Frame(left)
        canvas_frame.pack(fill="x")

        # Canvas to hold image thumbnails
        self.canvas = tk.Canvas(
            canvas_frame,
            height=120,
            bg="#E9F1F2",
            highlightthickness=0,
            xscrollincrement=1
        )
        self.canvas.pack(side="top", fill="x", expand=True)

        # Horizontal scrollbar for thumbnails
        x_scroll = ttk.Scrollbar(
            canvas_frame,
            orient="horizontal",
            command=self.canvas.xview,
            style="Grey.Horizontal.TScrollbar"
        )
        x_scroll.pack(side="bottom", fill="x")

        self.canvas.configure(xscrollcommand=x_scroll.set)

        # Enable horizontal scrolling using Shift + Mouse Wheel
        def _on_shift_mousewheel(event):
            # Windows uses event.delta for scroll amount
            self.canvas.xview_scroll(
                int(-1 * (event.delta / 120)),
                "units"
            )

        # Enable horizontal scrolling using Shift + Mouse Wheel
        self.canvas.bind("<Shift-MouseWheel>", _on_shift_mousewheel)

        # Enable click-and-drag scrolling on thumbnail canvas
        self.canvas.bind("<ButtonPress-1>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B1-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

        # Container to hold thumbnail frames inside the canvas
        self.thumb_container = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.thumb_container, anchor="nw")

        # Update scroll region whenever thumbnails change
        self.thumb_container.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Label shown when no image is selected
        self.preview_label = ttk.Label(
            left,
            text="Select an image from above",
            font=("Segoe UI", 20, "italic"),
            foreground="#1E2A2B",
        )
        self.preview_label.pack(expand=True, pady=10)

        # Text box to display OCR results and review cards
        self.ocr_box = tk.Text(
            right,
            wrap="word",
            bg="#fFFFFF",
            fg="#1E2A2B",
            insertbackground="#1E2A2B"
        )
        self.ocr_box.pack(fill="both", expand=True)

    # ===================== IMAGE ADDING =====================
    def add_images(self):
        # Open file dialog to select multiple image files
        paths = filedialog.askopenfilenames(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )

        for p in paths:
            # Skip image if already added
            if any(i["path"] == p for i in self.items):
                continue

            # Load image using PIL
            pil = Image.open(p).convert("RGB")

            # Create thumbnail for display
            thumb = pil.copy()
            thumb.thumbnail((160, 100))
            tkthumb = ImageTk.PhotoImage(thumb)

            idx = len(self.items)

            # Frame for thumbnail and filename
            frame = ttk.Frame(self.thumb_container, padding=4)
            lbl = ttk.Label(frame, image=tkthumb)
            lbl.image = tkthumb
            lbl.pack()

            # Click on thumbnail selects image
            lbl.bind("<Button-1>", lambda e, i=idx: self.select_image(i))

            # Display image filename
            ttk.Label(frame, text=os.path.basename(p), width=22).pack()
            frame.pack(side="left", padx=6)

            # Store image metadata
            self.items.append({
                "path": p,
                "pil": pil,
                "results": [],
                "frame": frame
            })

    # ===================== IMAGE SELECTION =====================
    def select_image(self, idx):
        self.selected_index = idx
        item = self.items[idx]

        # Clear previous preview
        for w in self.preview_label.winfo_children():
            w.destroy()

        # Resize image for preview display
        img = item["pil"].copy()
        img.thumbnail((520, 420))
        tkimg = ImageTk.PhotoImage(img)

        lbl = ttk.Label(self.preview_label, image=tkimg)
        lbl.image = tkimg
        lbl.pack(expand=True)

        # Display results related to this image
        self.show_results(item)

    # ===================== RESULT DISPLAY =====================
    def show_results(self, item):
        # Clear OCR text area
        self.ocr_box.config(state="normal")
        self.ocr_box.delete("1.0", "end")

        # Remove old results container if present
        if self.results_container:
            self.results_container.destroy()
            self.results_container = None

        # Create new container for result cards
        container = ttk.Frame(self.ocr_box)
        self.results_container = container
        self.ocr_box.window_create("end", window=container)

        self.ocr_box.config(state="disabled")

        # Create a result card for each detected review
        for r in item["results"]:
            card = ttk.Frame(container, style="Card.TFrame", padding=12)
            card.pack(fill="x", pady=8)

            conf = r.get("confidence", 50.0)

            # Display prediction and confidence
            ttk.Label(
                card,
                text=f"{r['prediction']} • {conf:.2f}%",
                style="CardTitle.TLabel"
            ).pack(anchor="w")

            meta = r.get("meta", {})

            # Display detailed feature scores
            ttk.Label(
                card,
                text=(
                    f"Generic: {meta.get('generic_score', 0)}% | "
                    f"Detail: {meta.get('detail_score', 0)}% | "
                    f"Sentiment: {meta.get('sentiment', 0)}% | "
                    f"Repetition: {meta.get('repetition_penalty', 0)}% | "
                    f"Promotional: {meta.get('promotional_score', 0)}%"
                ),
                style="CardText.TLabel"
            ).pack(anchor="w", pady=(4, 0))

            # Text widget to show review content
            text_widget = tk.Text(
                card,
                height=4,
                wrap="word",
                bg="#FFFFFF",
                fg="#1E2A2B",
                insertbackground="#1E2A2B",
                relief="flat"
            )
            text_widget.pack(fill="x", pady=(6, 0))

            text_widget.insert("1.0", r["text"])
            text_widget.tag_config("generic", foreground="#ffcc00")

            # Highlight generic phrases for non-genuine reviews
            if r["prediction"] != "Genuine":
                highlight_generic(
                    text_widget,
                    r["text"],
                    r["meta"].get("generic_phrases", [])
                )

            text_widget.config(state="disabled")

        self.ocr_box.config(state="disabled")

    # ===================== ANALYSIS PIPELINE =====================
    def start_analyze(self):
        # Ensure at least one image is added
        if not self.items:
            messagebox.showwarning("No images", "Add images first.")
            return

        # Run analysis in background thread
        threading.Thread(
            target=self._analyze_worker,
            daemon=True
        ).start()

    def _analyze_worker(self):
        try:
            # Show loader overlay
            self.root.after(
                0,
                lambda: self.loader.show("Extracting & analyzing...")
            )

            platform = self.platform_var.get()
            print("[DEBUG] Selected platform:", platform)

            # Process each uploaded image
            for it in self.items:
                # Keep UI responsive
                self.root.update_idletasks()

                it["results"] = []

                # ---------------- OCR STEP ----------------
                try:
                    merged_text = extract_text_from_image(it["path"])
                except Exception as e:
                    self.root.after(
                        0,
                        lambda err=e, p=it["path"]: messagebox.showerror(
                            "OCR Error",
                            f"OCR failed for image:\n{os.path.basename(p)}\n\n{err}"
                        )
                    )
                    continue

                print(f"[DEBUG] OCR text length: {len(merged_text)}")

                # ---------------- REVIEW SPLITTING ----------------
                review_blocks = split_reviews(merged_text, platform)

                print("\n========== REVIEW BLOCKS TO GUI ==========")
                for i, r in enumerate(review_blocks):
                    print(f"GUI BLOCK {i}: {repr(r)}")
                print("=========================================\n")

                # ---------------- DEDUPLICATION ----------------
                seen_reviews = set()

                for idx, review in enumerate(review_blocks):

                    print(f"\n[PIPELINE] ===== REVIEW {idx} START =====")
                    print("[PIPELINE] RAW REVIEW:", repr(review))

                    raw_text = review.strip()

                    # ---------- Valid review filter ----------
                    if not is_valid_review(raw_text):
                        print("[PIPELINE] ❌ REJECTED by is_valid_review()")
                        continue

                    print("[PIPELINE] ✅ Passed is_valid_review")

                    # Use raw text for display and cleaned text for model input
                    display_text = raw_text
                    model_text = clean_text(display_text)

                    # Debug logs for tracing text transformation
                    print("[PIPELINE] DISPLAY TEXT:", repr(display_text))
                    print("[PIPELINE] MODEL TEXT:", repr(model_text))
                    print("[PIPELINE] MODEL WORD COUNT:", len(model_text.split()))

                    # ---------- Word length filter ----------
                    # Flipkart reviews are often very short, so allow fewer words
                    min_words = 1 if platform.lower() == "flipkart" else 3

                    if len(model_text.split()) < min_words:
                        print("[PIPELINE] ❌ REJECTED by word-count filter (<{min_words})")
                        continue
                    print("[PIPELINE] ✅ Passed word-count filter")

                    # ---------- Deduplication ----------
                    # Hash cleaned text to avoid analyzing duplicate reviews
                    fingerprint = hashlib.md5(
                        model_text.encode("utf-8")
                    ).hexdigest()

                    if fingerprint in seen_reviews:
                        print("[PIPELINE] ❌ REJECTED as DUPLICATE")
                        continue

                    seen_reviews.add(fingerprint)
                    print("[PIPELINE] ✅ Passed deduplication")

                    # ---------- ML Model Prediction ----------
                    pred, conf, meta = analyze_review(display_text, platform)

                    print("[PIPELINE] ✅ MODEL ACCEPTED REVIEW")
                    print("[PIPELINE] PRED:", pred, "| CONF:", conf)

                    # Store result for this review
                    it["results"].append({
                        "text": display_text,
                        "prediction": pred,
                        "confidence": round(conf, 2),
                        "meta": meta
                    })

                    print(f"[PIPELINE] ===== REVIEW {idx} END =====\n")

        finally:
            # Always hide loader even if an error occurs
            self.root.after(0, self._finish)

    # ===================== FINAL SUMMARY =====================
    def _finish(self):
        # Hide loader overlay
        self.loader.hide()

        # Count reviews by category
        total_reviews = sum(len(i["results"]) for i in self.items)

        likely_fake = sum(
            1 for i in self.items for r in i["results"]
            if r["prediction"] == "Likely Fake"
        )
        genuine = sum(
            1 for i in self.items for r in i["results"]
            if r["prediction"] == "Genuine"
        )
        suspicious = sum(
            1 for i in self.items for r in i["results"]
            if r["prediction"] == "Suspicious"
        )
        neutral = sum(
            1 for i in self.items for r in i["results"]
            if r["prediction"] == "Neutral"
        )

        # Display summary in OCR text box
        self.ocr_box.config(state="normal")
        self.ocr_box.delete("1.0", "end")
        self.ocr_box.insert(
            "1.0",
            f"Images analyzed: {len(self.items)}\n"
            f"Total reviews extracted: {total_reviews}\n"
            f"Likely fake reviews: {likely_fake}\n"
            f"Genuine reviews: {genuine}\n"
            f"Suspicious reviews: {suspicious}\n"
            f"Neutral reviews: {neutral}\n\n"
            "Click an image to view detailed cards."
        )
        self.ocr_box.config(state="disabled")

    # ===================== CHART VISUALIZATION =====================
    def show_charts(self):
        # Ensure an image is selected
        if self.selected_index is None:
            messagebox.showinfo("Select image", "Select an image first.")
            return

        results = self.items[self.selected_index]["results"]

        # No data to plot
        if not results:
            messagebox.showinfo("No data", "No analyzed reviews to plot.")
            return

        chart = self.chart_type.get()

        # Plot summary distribution chart
        if chart == "Summary":
            charts.plot_review_summary(results)

        # Plot average confidence chart
        elif chart == "Average Confidence":
            charts.plot_avg_confidence(results)

    # ===================== CLEAR ALL DATA =====================
    def clear_images(self):
        # Remove thumbnail widgets
        for it in self.items:
            it["frame"].destroy()

        # Reset application state
        self.items.clear()
        self.selected_index = None

        # Reset preview panel
        for w in self.preview_label.winfo_children():
            w.destroy()
        self.preview_label.config(text="Select an image from above")

        # Clear OCR output area
        self.ocr_box.config(state="normal")
        self.ocr_box.delete("1.0", "end")

        if self.results_container:
            self.results_container.destroy()
            self.results_container = None

        self.ocr_box.config(state="disabled")


# ===================== APPLICATION ENTRY POINT =====================
if __name__ == "__main__":
    root = tk.Tk()

    # Load Azure theme for consistent UI styling
    root.tk.call("source", "gui/theme/azure/azure.tcl")
    root.tk.call("set_theme", "light")

    # Launch application
    App(root)
    root.mainloop()