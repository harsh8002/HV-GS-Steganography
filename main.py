import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import pandas as pd
import joblib
import threading
import random
import os
import sys
import os
import lightgbm

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ---------------- THEME ----------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ---------------- COLORS ----------------
BG_DARK = "#0B1120"
CARD = "#111827"
BORDER = "#1F2937"
INPUT_BG = "#020617"
PRIMARY = "#3B82F6"
PRIMARY_HOVER = "#2563EB"
SUCCESS = "#22C55E"
SUCCESS_HOVER = "#16A34A"
TEXT_SOFT = "#94A3B8"
BTN_BG = "#1F2937"
BTN_HOVER = "#334155"

TOGGLE_ACTIVE = PRIMARY
TOGGLE_INACTIVE = BTN_BG
TOGGLE_HOVER = BTN_HOVER

# ---------------- MODELS ----------------
models = {
    "LightGBM": joblib.load(resource_path("channel_selector_model.pkl")),
    "Decision Tree": joblib.load(resource_path("channel_model.pkl"))
}

FEATURE_COLUMNS = ["R", "G", "B", "Y", "variance"]


def get_features(pixel, image, i, j):
    R, G, B = pixel
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    patch = image[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
    variance = np.var(patch)
    return [R, G, B, Y, variance]


def safe_output_name(name: str, default_name: str) -> str:
    name = (name or "").strip()
    if not name:
        name = default_name
    name = os.path.basename(name)
    _, ext = os.path.splitext(name)
    if not ext:
        name = name + (".png" if default_name.lower().endswith(".png") else ".txt")
    return name


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("HV-GS Steganography Tool")
        self.geometry("1120x620")
        self.minsize(1000, 560)
        self.configure(fg_color=BG_DARK)

        self.image_path = ""
        self.payload_path = ""
        self.save_path = ""
        self.mode = "hide"
        self.selected_model = "LightGBM"
        self.processing = False

        # ---------------- OUTER LAYOUT ----------------
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main_frame = ctk.CTkScrollableFrame(
            self,
            corner_radius=18,
            fg_color=CARD,
            border_width=1,
            border_color=BORDER
        )
        self.main_frame.grid(row=0, column=0, padx=16, pady=16, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.side_panel = ctk.CTkFrame(
            self,
            corner_radius=18,
            fg_color=CARD,
            border_width=1,
            border_color=BORDER
        )
        self.side_panel.grid(row=0, column=1, padx=(0, 16), pady=16, sticky="nsew")
        self.side_panel.grid_columnconfigure(0, weight=1)

        # ---------------- TITLE ----------------
        ctk.CTkLabel(
            self.main_frame,
            text="HV-GS Steganography",
            font=("Segoe UI", 24, "bold")
        ).grid(row=0, column=0, pady=(10, 2), padx=16, sticky="n")

        ctk.CTkLabel(
            self.main_frame,
            text="Adaptive ML-based image steganography with key-controlled randomization",
            font=("Segoe UI", 11),
            text_color=TEXT_SOFT
        ).grid(row=1, column=0, pady=(0, 6), padx=16, sticky="n")

        # ---------------- MODE LABEL ----------------
        ctk.CTkLabel(
            self.main_frame,
            text="Mode",
            font=("Segoe UI", 12, "bold"),
            text_color=TEXT_SOFT
        ).grid(row=2, column=0, pady=(0, 4), padx=16, sticky="w")

        # ---------------- MODE SEGMENTED CONTROL ----------------
        self.mode_segment = ctk.CTkFrame(
            self.main_frame,
            corner_radius=15,
            fg_color=BTN_BG,
            border_width=1,
            border_color=BORDER
        )
        self.mode_segment.grid(row=3, column=0, pady=(0, 8), padx=16, sticky="w")

        self.mode_buttons = {}

        self.mode_buttons["hide"] = ctk.CTkButton(
            self.mode_segment,
            text="◉ Hide",
            width=150,
            height=36,
            corner_radius=0,
            fg_color=TOGGLE_INACTIVE,
            hover_color=TOGGLE_HOVER,
            text_color=TEXT_SOFT,
            font=("Segoe UI", 12, "bold"),
            command=lambda: self.switch_mode("hide")
        )
        self.mode_buttons["hide"].grid(row=0, column=0, sticky="nsew", padx=1, pady=1)

        self.mode_buttons["extract"] = ctk.CTkButton(
            self.mode_segment,
            text="◉ Extract",
            width=150,
            height=36,
            corner_radius=0,
            fg_color=TOGGLE_INACTIVE,
            hover_color=TOGGLE_HOVER,
            text_color=TEXT_SOFT,
            font=("Segoe UI", 12, "bold"),
            command=lambda: self.switch_mode("extract")
        )
        self.mode_buttons["extract"].grid(row=0, column=1, sticky="nsew", padx=(0, 1), pady=1)

        # ---------------- INPUT CARD ----------------
        self.input_card = ctk.CTkFrame(
            self.main_frame,
            corner_radius=16,
            fg_color=INPUT_BG,
            border_width=1,
            border_color=BORDER
        )
        self.input_card.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 8))
        self.input_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.input_card,
            text="Input Settings",
            font=("Segoe UI", 15, "bold")
        ).grid(row=0, column=0, pady=(8, 4), padx=12, sticky="w")

        self.img_entry = ctk.CTkEntry(
            self.input_card,
            height=34,
            corner_radius=10,
            fg_color=CARD,
            border_color=BORDER,
            placeholder_text="Select Image..."
        )
        self.img_entry.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 4))

        self.image_browse_btn = ctk.CTkButton(
            self.input_card,
            text="Browse Image",
            height=30,
            corner_radius=10,
            fg_color=BTN_BG,
            hover_color=BTN_HOVER,
            border_width=1,
            border_color=BORDER,
            font=("Segoe UI", 10),
            command=self.select_image
        )
        self.image_browse_btn.grid(row=2, column=0, pady=(0, 8), padx=12, sticky="w")

        self.key_entry = ctk.CTkEntry(
            self.input_card,
            height=34,
            corner_radius=10,
            fg_color=CARD,
            border_color=BORDER,
            placeholder_text="Enter Secret Key"
        )
        self.key_entry.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 6))

        ctk.CTkLabel(
            self.input_card,
            text="Select Model",
            font=("Segoe UI", 13, "bold")
        ).grid(row=4, column=0, pady=(0, 4), padx=12, sticky="w")

        # ---------------- MODEL SEGMENTED CONTROL ----------------
        self.model_segment = ctk.CTkFrame(
            self.input_card,
            corner_radius=8,
            fg_color=BTN_BG,
            border_width=1,
            border_color=BORDER
        )
        self.model_segment.grid(row=5, column=0, pady=(0, 8), padx=12, sticky="w")

        self.model_buttons = {}
        self.model_buttons["LightGBM"] = ctk.CTkButton(
            self.model_segment,
            text="LightGBM",
            width=132,
            height=30,
            corner_radius=0,
            fg_color=TOGGLE_INACTIVE,
            hover_color=TOGGLE_HOVER,
            text_color=TEXT_SOFT,
            font=("Segoe UI", 10),
            command=lambda: self.select_model("LightGBM")
        )
        self.model_buttons["LightGBM"].grid(row=0, column=0, sticky="nsew", padx=1, pady=1)

        self.model_buttons["Decision Tree"] = ctk.CTkButton(
            self.model_segment,
            text="Decision Tree",
            width=132,
            height=30,
            corner_radius=0,
            fg_color=TOGGLE_INACTIVE,
            hover_color=TOGGLE_HOVER,
            text_color=TEXT_SOFT,
            font=("Segoe UI", 10),
            command=lambda: self.select_model("Decision Tree")
        )
        self.model_buttons["Decision Tree"].grid(row=0, column=1, sticky="nsew", padx=(0, 1), pady=1)

        # ---------------- HIDE ONLY CARD ----------------
        self.hide_card = ctk.CTkFrame(
            self.main_frame,
            corner_radius=16,
            fg_color=INPUT_BG,
            border_width=1,
            border_color=BORDER
        )
        self.hide_card.grid(row=5, column=0, sticky="ew", padx=16, pady=(0, 8))
        self.hide_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.hide_card,
            text="Hide Settings",
            font=("Segoe UI", 15, "bold")
        ).grid(row=0, column=0, pady=(8, 4), padx=12, sticky="w")

        self.payload_entry = ctk.CTkEntry(
            self.hide_card,
            height=34,
            corner_radius=10,
            fg_color=CARD,
            border_color=BORDER,
            placeholder_text="Select Payload Text File (.txt)"
        )
        self.payload_entry.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 4))

        self.payload_browse_btn = ctk.CTkButton(
            self.hide_card,
            text="Browse Payload",
            height=30,
            corner_radius=10,
            fg_color=BTN_BG,
            hover_color=BTN_HOVER,
            border_width=1,
            border_color=BORDER,
            font=("Segoe UI", 10),
            command=self.select_payload
        )
        self.payload_browse_btn.grid(row=2, column=0, pady=(0, 6), padx=12, sticky="w")

        ctk.CTkLabel(
            self.hide_card,
            text="Output filename is saved in the same folder as the input image.",
            font=("Segoe UI", 10),
            text_color=TEXT_SOFT,
            justify="center"
        ).grid(row=3, column=0, padx=12, pady=(0, 4), sticky="ew")

        self.output_entry = ctk.CTkEntry(
            self.hide_card,
            height=34,
            corner_radius=10,
            fg_color=CARD,
            border_color=BORDER,
            placeholder_text="Output filename (example: stego.png)"
        )
        self.output_entry.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 8))

        # ---------------- RESULT CARD ----------------
        self.result_card = ctk.CTkFrame(
            self.main_frame,
            corner_radius=16,
            fg_color=INPUT_BG,
            border_width=1,
            border_color=BORDER
        )
        self.result_card.grid(row=6, column=0, sticky="nsew", padx=16, pady=(0, 8))
        self.result_card.grid_columnconfigure(0, weight=1)
        self.result_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self.result_card,
            text="Extracted Data",
            font=("Segoe UI", 15, "bold")
        ).grid(row=0, column=0, pady=(8, 4), padx=12, sticky="w")

        self.result_textbox = ctk.CTkTextbox(
            self.result_card,
            height=120,
            corner_radius=10,
            fg_color=CARD,
            border_color=BORDER,
            border_width=1,
            wrap="word"
        )
        self.result_textbox.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 6))
        self._set_result_text("Extracted text will appear here.")
        self.result_textbox.configure(state="disabled")

        self.copy_btn = ctk.CTkButton(
            self.result_card,
            text="Copy Extracted Text",
            width=160,
            height=30,
            corner_radius=10,
            fg_color=BTN_BG,
            hover_color=BTN_HOVER,
            border_width=1,
            border_color=BORDER,
            font=("Segoe UI", 10),
            command=self.copy_extracted_text
        )
        self.copy_btn.grid(row=2, column=0, pady=(0, 8))

        self.result_card.grid_remove()

        # ---------------- ACTION CARD ----------------
        self.action_card = ctk.CTkFrame(
            self.main_frame,
            corner_radius=16,
            fg_color=INPUT_BG,
            border_width=1,
            border_color=BORDER
        )
        self.action_card.grid(row=7, column=0, sticky="ew", padx=16, pady=(0, 8))
        self.action_card.grid_columnconfigure(0, weight=1)

        self.progress = ctk.CTkProgressBar(self.action_card)
        self.progress.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 3))
        self.progress.configure(progress_color=PRIMARY, fg_color=BORDER, height=9, corner_radius=10)
        self.progress.set(0)

        self.progress_label = ctk.CTkLabel(self.action_card, text="0%", font=("Segoe UI", 10))
        self.progress_label.grid(row=1, column=0, pady=(0, 4))

        self.run_btn = ctk.CTkButton(
            self.action_card,
            text="Hide Data",
            height=38,
            corner_radius=14,
            fg_color=SUCCESS,
            hover_color=SUCCESS_HOVER,
            font=("Segoe UI", 13, "bold"),
            command=self.start_process
        )
        self.run_btn.grid(row=2, column=0, padx=12, pady=(0, 10))

        # ---------------- SIDE PANEL ----------------
        ctk.CTkLabel(
            self.side_panel,
            text="Payload Calculator",
            font=("Segoe UI", 16, "bold")
        ).grid(row=0, column=0, pady=(12, 6), padx=12, sticky="n")

        self.width_entry = ctk.CTkEntry(
            self.side_panel,
            height=34,
            corner_radius=10,
            fg_color=CARD,
            border_color=BORDER,
            placeholder_text="Enter Width"
        )
        self.width_entry.grid(row=1, column=0, padx=12, pady=(0, 4), sticky="ew")

        self.height_entry = ctk.CTkEntry(
            self.side_panel,
            height=34,
            corner_radius=10,
            fg_color=CARD,
            border_color=BORDER,
            placeholder_text="Enter Height"
        )
        self.height_entry.grid(row=2, column=0, padx=12, pady=(0, 6), sticky="ew")

        ctk.CTkLabel(
            self.side_panel,
            text="Select an image or enter width & height\nfor quick capacity estimation.",
            font=("Segoe UI", 10),
            text_color=TEXT_SOFT,
            justify="center"
        ).grid(row=3, column=0, padx=12, pady=(0, 6), sticky="ew")

        self.result_label = ctk.CTkLabel(
            self.side_panel,
            text="Max English characters that can be embedded:\n-",
            font=("Segoe UI", 14, "bold")
        )
        self.result_label.grid(row=4, column=0, padx=12, pady=(0, 8), sticky="n")

        ctk.CTkLabel(
            self.side_panel,
            text="Formula:\nBits = W × H × 1\nChars = Bits / 8\n\n1 LSB per selected channel",
            justify="left",
            text_color=TEXT_SOFT
        ).grid(row=5, column=0, padx=12, pady=(0, 10), sticky="nw")

        self.width_entry.bind("<KeyRelease>", self.calculate_payload)
        self.height_entry.bind("<KeyRelease>", self.calculate_payload)

        self.switch_mode("hide")
        self.apply_toggle_state("hide", self.mode_buttons)
        self.apply_toggle_state("LightGBM", self.model_buttons)

    # ---------------- TOGGLE HELPERS ----------------
    def apply_toggle_state(self, active_key, buttons_dict):
        for key, btn in buttons_dict.items():
            if key == active_key:
                btn.configure(
                    fg_color=TOGGLE_ACTIVE,
                    hover_color=TOGGLE_ACTIVE,
                    text_color="white"
                )
            else:
                btn.configure(
                    fg_color=TOGGLE_INACTIVE,
                    hover_color=TOGGLE_HOVER,
                    text_color=TEXT_SOFT
                )

    # ---------------- UI HELPERS ----------------
    def _set_progress(self, value: float):
        value = max(0.0, min(1.0, float(value)))

        def _do():
            self.progress.set(value)
            self.progress_label.configure(text=f"{int(value * 100)}%")

        self.after(0, _do)

    def _set_run_button(self, enabled: bool, text: str):
        def _do():
            self.run_btn.configure(state="normal" if enabled else "disabled", text=text)

        self.after(0, _do)

    def _show_info(self, title: str, text: str):
        self.after(0, lambda: messagebox.showinfo(title, text))

    def _show_error(self, title: str, text: str):
        self.after(0, lambda: messagebox.showerror(title, text))

    def _set_result_text(self, text: str):
        def _do():
            self.result_textbox.configure(state="normal")
            self.result_textbox.delete("1.0", "end")
            self.result_textbox.insert("1.0", text)
            self.result_textbox.configure(state="disabled")

        self.after(0, _do)

    def _reset_action(self):
        self.processing = False
        self._set_run_button(True, "Hide Data" if self.mode == "hide" else "Extract Data")
        self._set_progress(0.0)

    # ---------------- MODEL SELECT ----------------
    def select_model(self, name):
        self.selected_model = name
        self.apply_toggle_state(name, self.model_buttons)

    # ---------------- MODE SWITCH ----------------
    def switch_mode(self, mode):
        self.mode = mode
        self.apply_toggle_state(mode, self.mode_buttons)

        if mode == "hide":
            self.hide_card.grid()
            self.result_card.grid_remove()
            self.run_btn.configure(text="Hide Data", fg_color=SUCCESS, hover_color=SUCCESS_HOVER)
        else:
            self.hide_card.grid_remove()
            self.result_card.grid()
            self._set_result_text("Extracted text will appear here.")
            self.run_btn.configure(text="Extract Data", fg_color=PRIMARY, hover_color=PRIMARY_HOVER)

    # ---------------- FILE SELECT ----------------
    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")]
        )
        if not path:
            return

        self.image_path = path
        self.img_entry.delete(0, "end")
        self.img_entry.insert(0, path)

        img = cv2.imread(path)
        if img is None:
            self._show_error("Error", "Could not read the selected image.")
            return

        h, w, _ = img.shape
        self.width_entry.delete(0, "end")
        self.width_entry.insert(0, str(w))
        self.height_entry.delete(0, "end")
        self.height_entry.insert(0, str(h))
        self.calculate_payload()

        base = os.path.splitext(os.path.basename(path))[0]
        suggested = f"{base}_stego.png"
        current_out = self.output_entry.get().strip()
        if not current_out or current_out == "stego.png":
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, suggested)

    def select_payload(self):
        path = filedialog.askopenfilename(
            title="Select Payload Text File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not path:
            return

        self.payload_path = path
        self.payload_entry.delete(0, "end")
        self.payload_entry.insert(0, path)

    # ---------------- START PROCESS ----------------
    def start_process(self):
        if self.processing:
            return

        if not self.image_path:
            messagebox.showerror("Error", "Please select an image first.")
            return

        if not self.key_entry.get().strip():
            messagebox.showerror("Error", "Please enter a secret key.")
            return

        if self.mode == "hide" and not self.payload_path:
            messagebox.showerror("Error", "Please select a payload text file.")
            return

        if self.mode == "hide":
            out_name = safe_output_name(self.output_entry.get(), "stego.png")
            image_dir = os.path.dirname(self.image_path)
            self.save_path = os.path.join(image_dir, out_name)
        else:
            self.save_path = None

        self.processing = True
        self._set_progress(0.0)
        self._set_run_button(False, "Processing...")

        if self.mode == "hide":
            threading.Thread(target=self.embed, daemon=True).start()
        else:
            threading.Thread(target=self.extract, daemon=True).start()

    # ---------------- EMBED ----------------
    def embed(self):
        try:
            model = models[self.selected_model]

            img_bgr = cv2.imread(self.image_path)
            if img_bgr is None:
                raise ValueError("Invalid input image.")

            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            with open(self.payload_path, "rb") as f:
                data = f.read()

            header = format(len(data), "032b")
            payload_bits = "".join(format(b, "08b") for b in data)
            bits = header + payload_bits

            h, w, _ = img.shape

            seed = sum(ord(c) for c in self.key_entry.get().strip())
            random.seed(seed)

            coords = [(i, j) for i in range(h) for j in range(w)]
            random.shuffle(coords)

            if len(bits) > len(coords):
                raise ValueError("Payload too large for the selected image.")

            for idx, (i, j) in enumerate(coords[:len(bits)]):
                pixel = img[i, j]

                features = pd.DataFrame(
                    [get_features(pixel, img, i, j)],
                    columns=FEATURE_COLUMNS
                )

                ch = int(model.predict(features)[0])
                bit = int(bits[idx])

                img[i, j, ch] = (int(img[i, j, ch]) & 0xFE) | bit

                if idx % 2000 == 0 or idx == len(bits) - 1:
                    self._set_progress((idx + 1) / len(bits))

            out_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(self.save_path, out_bgr):
                raise ValueError("Could not save stego image.")

            self._show_info("Done", f"Stego image saved:\n{self.save_path}")
            self.after(0, self._reset_action)

        except Exception as e:
            self.after(0, self._reset_action)
            self._show_error("Error", str(e))

    # ---------------- EXTRACT ----------------
    def extract(self):
        try:
            model = models[self.selected_model]

            img_bgr = cv2.imread(self.image_path)
            if img_bgr is None:
                raise ValueError("Invalid stego image.")

            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape

            seed = sum(ord(c) for c in self.key_entry.get().strip())
            random.seed(seed)

            coords = [(i, j) for i in range(h) for j in range(w)]
            random.shuffle(coords)

            bits_collected = []
            target_total_bits = None

            for idx, (i, j) in enumerate(coords):
                pixel = img[i, j]

                features = pd.DataFrame(
                    [get_features(pixel, img, i, j)],
                    columns=FEATURE_COLUMNS
                )

                ch = int(model.predict(features)[0])
                bits_collected.append(str(int(img[i, j, ch]) & 1))

                if len(bits_collected) == 32 and target_total_bits is None:
                    length_bytes = int("".join(bits_collected[:32]), 2)
                    target_total_bits = 32 + (length_bytes * 8)

                    if target_total_bits > len(coords):
                        raise ValueError(
                            "Invalid key/model or corrupted stego image "
                            "(payload length exceeds capacity)."
                        )

                if target_total_bits is not None:
                    self._set_progress(len(bits_collected) / target_total_bits)
                    if len(bits_collected) >= target_total_bits:
                        break
                else:
                    self._set_progress(min(len(bits_collected) / 32.0, 0.35))

            if len(bits_collected) < 32:
                raise ValueError("Could not read payload header.")

            if target_total_bits is None:
                raise ValueError("Could not determine payload length.")

            data_bits = "".join(bits_collected[32:target_total_bits])

            byte_data = bytearray()
            for i in range(0, len(data_bits), 8):
                byte_chunk = data_bits[i:i + 8]
                if len(byte_chunk) < 8:
                    break
                byte_data.append(int(byte_chunk, 2))

            extracted_text = byte_data.decode("utf-8", errors="replace").rstrip("\x00")
            self._set_result_text(extracted_text)
            self._show_info("Done", "Text extracted and shown in the UI.")
            self.after(0, self._reset_action)

        except Exception as e:
            self.after(0, self._reset_action)
            self._show_error("Error", str(e))

    # ---------------- COPY EXTRACTED TEXT ----------------
    def copy_extracted_text(self):
        try:
            self.result_textbox.configure(state="normal")
            text = self.result_textbox.get("1.0", "end").strip()
            self.result_textbox.configure(state="disabled")

            self.clipboard_clear()
            self.clipboard_append(text)
            self._show_info("Copied", "Extracted text copied to clipboard.")
        except Exception as e:
            self._show_error("Error", str(e))

    # ---------------- PAYLOAD CALCULATOR ----------------
    def calculate_payload(self, event=None):
        try:
            w = int(self.width_entry.get())
            h = int(self.height_entry.get())

            total_bits = w * h * 1
            total_chars = total_bits // 8

            self.result_label.configure(text=f"Max English chars:\n{total_chars:,}")
        except Exception:
            self.result_label.configure(text="Max English chars:\n-")


if __name__ == "__main__":
    app = App()
    app.mainloop()
