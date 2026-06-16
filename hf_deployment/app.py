"""
Aplikasi Prediksi Harga Laptop
Deployed on Hugging Face Spaces menggunakan Gradio
"""

import gradio as gr
import numpy as np
import joblib
import json
import os

# ── Load Model & Artifacts ─────────────────────────────────────────────────────
model = joblib.load("laptop_price_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

with open("model_metadata.json", "r") as f:
    metadata = json.load(f)

feature_cols = metadata["feature_cols"]

# ── Opsi Pilihan ───────────────────────────────────────────────────────────────
BRANDS = sorted(['Asus', 'Lenovo', 'HP', 'Dell', 'Acer', 'MSI', 'Apple', 'Samsung', 'Razer', 'Toshiba'])
PROCESSORS = sorted([
    'Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Intel Core i9',
    'AMD Ryzen 3', 'AMD Ryzen 5', 'AMD Ryzen 7', 'AMD Ryzen 9',
    'Apple M1', 'Apple M2'
])
STORAGE_TYPES = ['HDD', 'SSD', 'NVMe SSD']
GPUS = sorted([
    'Integrated', 'NVIDIA GTX 1650', 'NVIDIA RTX 3050', 'NVIDIA RTX 3060',
    'NVIDIA RTX 3070', 'NVIDIA RTX 4060', 'AMD Radeon RX 6600', 'Apple GPU'
])
OS_OPTIONS = ['Windows 11', 'Windows 10', 'macOS', 'Linux', 'Chrome OS']
SCREEN_SIZES = [11.6, 13.3, 14.0, 15.6, 16.0, 17.3]
REFRESH_RATES = [60, 90, 120, 144, 165, 240]
RAM_OPTIONS = [4, 8, 16, 32, 64]
STORAGE_OPTIONS = [128, 256, 512, 1024, 2048]

GAMING_GPUS = ['NVIDIA GTX 1650', 'NVIDIA RTX 3050', 'NVIDIA RTX 3060',
               'NVIDIA RTX 3070', 'NVIDIA RTX 4060', 'AMD Radeon RX 6600']


def format_idr(amount):
    """Format angka ke format Rupiah."""
    if amount >= 1_000_000:
        return f"Rp {amount/1_000_000:.1f} Juta"
    return f"Rp {amount:,.0f}"


def predict_price(brand, processor, ram_gb, storage_gb, storage_type,
                  gpu, screen_size, refresh_rate, battery_wh,
                  weight_kg, has_touchscreen, os_type):
    """Prediksi harga laptop berdasarkan spesifikasi."""
    try:
        # ── Feature Engineering ────────────────────────────────────────────────
        storage_per_ram = storage_gb / ram_gb
        compute_score = ram_gb * (storage_gb / 512)
        is_gaming = 1 if gpu in GAMING_GPUS else 0

        # ── Label Encoding ─────────────────────────────────────────────────────
        cat_values = {
            'brand': brand,
            'processor': processor,
            'storage_type': storage_type,
            'gpu': gpu,
            'os': os_type
        }

        encoded = {}
        for col, val in cat_values.items():
            le = label_encoders[col]
            if val in le.classes_:
                encoded[col] = le.transform([val])[0]
            else:
                encoded[col] = 0  # fallback

        # ── Build Feature Array ────────────────────────────────────────────────
        features = np.array([[
            encoded['brand'],
            encoded['processor'],
            ram_gb,
            storage_gb,
            encoded['storage_type'],
            encoded['gpu'],
            screen_size,
            refresh_rate,
            battery_wh,
            weight_kg,
            int(has_touchscreen),
            encoded['os'],
            storage_per_ram,
            compute_score,
            is_gaming
        ]])

        # ── Predict (log scale → original) ────────────────────────────────────
        log_pred = model.predict(features)[0]
        price = np.expm1(log_pred)

        # Rentang harga (± 10%)
        price_low = price * 0.90
        price_high = price * 1.10

        # Kategori harga
        if price < 7_000_000:
            category = "💰 Budget"
            cat_color = "#4CAF50"
        elif price < 15_000_000:
            category = "⚡ Mid-Range"
            cat_color = "#2196F3"
        elif price < 25_000_000:
            category = "🔥 High-End"
            cat_color = "#FF9800"
        else:
            category = "👑 Premium"
            cat_color = "#9C27B0"

        # ── Output HTML ────────────────────────────────────────────────────────
        output_html = f"""
        <div style="font-family: 'Segoe UI', sans-serif; max-width: 560px; margin: 0 auto;">
          <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                      border-radius: 16px; padding: 24px; color: white; text-align: center;
                      box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <div style="font-size: 15px; opacity: 0.8; margin-bottom: 6px;">
              💻 {brand} — {processor}
            </div>
            <div style="font-size: 42px; font-weight: 800; color: #4fc3f7; margin: 8px 0;">
              {format_idr(price)}
            </div>
            <div style="font-size: 14px; opacity: 0.7; margin-top: 4px;">
              Estimasi: {format_idr(price_low)} — {format_idr(price_high)}
            </div>
            <div style="display: inline-block; margin-top: 12px;
                        background: {cat_color}22; border: 2px solid {cat_color};
                        color: {cat_color}; padding: 6px 20px; border-radius: 50px;
                        font-size: 15px; font-weight: 600;">
              {category}
            </div>
          </div>

          <div style="background: #f8f9fa; border-radius: 12px; padding: 18px; margin-top: 16px;">
            <div style="font-weight: 700; font-size: 14px; color: #333; margin-bottom: 12px;
                        text-transform: uppercase; letter-spacing: 1px;">
              📋 Spesifikasi yang Diinput
            </div>
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
              <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 7px 4px; color: #666; width: 45%;">Merek</td>
                <td style="padding: 7px 4px; font-weight: 600; color: #222;">{brand}</td>
              </tr>
              <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 7px 4px; color: #666;">Prosesor</td>
                <td style="padding: 7px 4px; font-weight: 600; color: #222;">{processor}</td>
              </tr>
              <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 7px 4px; color: #666;">RAM</td>
                <td style="padding: 7px 4px; font-weight: 600; color: #222;">{ram_gb} GB</td>
              </tr>
              <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 7px 4px; color: #666;">Storage</td>
                <td style="padding: 7px 4px; font-weight: 600; color: #222;">{storage_gb} GB {storage_type}</td>
              </tr>
              <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 7px 4px; color: #666;">GPU</td>
                <td style="padding: 7px 4px; font-weight: 600; color: #222;">{gpu}</td>
              </tr>
              <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 7px 4px; color: #666;">Layar</td>
                <td style="padding: 7px 4px; font-weight: 600; color: #222;">{screen_size}" — {refresh_rate}Hz</td>
              </tr>
              <tr>
                <td style="padding: 7px 4px; color: #666;">Sistem Operasi</td>
                <td style="padding: 7px 4px; font-weight: 600; color: #222;">{os_type}</td>
              </tr>
            </table>
          </div>

          <div style="background: #e3f2fd; border-left: 4px solid #2196F3;
                      border-radius: 8px; padding: 12px 16px; margin-top: 12px;
                      font-size: 12px; color: #555;">
            ⚠️ <strong>Catatan:</strong> Prediksi ini merupakan estimasi berdasarkan model
            Random Forest dengan R² ≈ {metadata['metrics']['R2']:.3f}.
            Harga aktual di pasaran dapat berbeda ±10–15%.
          </div>
        </div>
        """
        return output_html

    except Exception as e:
        return f"<div style='color:red; padding:16px;'>⚠️ Error: {str(e)}</div>"


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Prediksi Harga Laptop",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
    css="""
        .gradio-container { max-width: 900px !important; margin: auto; }
        .gr-button-primary { background: linear-gradient(135deg, #1a73e8, #0d47a1) !important; }
        footer { display: none !important; }
    """
) as demo:

    gr.Markdown("""
    # 💻 Prediksi Harga Laptop
    ### Masukkan spesifikasi laptop untuk mendapatkan estimasi harga pasar Indonesia (IDR)
    *Model: Random Forest | Dataset: 1.500 data laptop | Akurasi R² ≥ 0.90*
    ---
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🏷️ Identitas Laptop")
            brand = gr.Dropdown(BRANDS, label="Merek (Brand)", value="Lenovo")
            os_type = gr.Dropdown(OS_OPTIONS, label="Sistem Operasi", value="Windows 11")
            processor = gr.Dropdown(PROCESSORS, label="Prosesor", value="Intel Core i5")

            gr.Markdown("### 💾 Memori & Penyimpanan")
            ram_gb = gr.Dropdown([str(r) for r in RAM_OPTIONS], label="RAM (GB)", value="8")
            storage_gb = gr.Dropdown([str(s) for s in STORAGE_OPTIONS], label="Storage (GB)", value="512")
            storage_type = gr.Dropdown(STORAGE_TYPES, label="Tipe Storage", value="SSD")

        with gr.Column(scale=1):
            gr.Markdown("### 🖥️ Layar & Grafis")
            screen_size = gr.Dropdown([str(s) for s in SCREEN_SIZES], label="Ukuran Layar (inch)", value="15.6")
            refresh_rate = gr.Dropdown([str(r) for r in REFRESH_RATES], label="Refresh Rate (Hz)", value="60")
            gpu = gr.Dropdown(GPUS, label="GPU / Kartu Grafis", value="Integrated")
            has_touchscreen = gr.Checkbox(label="Touchscreen", value=False)

            gr.Markdown("### 🔋 Fisik & Baterai")
            battery_wh = gr.Slider(30, 100, value=50, step=1, label="Kapasitas Baterai (Wh)")
            weight_kg = gr.Slider(0.9, 3.5, value=1.8, step=0.1, label="Berat Laptop (kg)")

    predict_btn = gr.Button("🔮 Prediksi Harga Sekarang!", variant="primary", size="lg")

    output_html = gr.HTML(label="Hasil Prediksi")

    predict_btn.click(
        fn=lambda b, p, r, s, st, g, ss, rr, bw, w, t, o: predict_price(
            b, p, int(r), int(s), st, g, float(ss), int(rr), bw, w, t, o
        ),
        inputs=[brand, processor, ram_gb, storage_gb, storage_type,
                gpu, screen_size, refresh_rate, battery_wh,
                weight_kg, has_touchscreen, os_type],
        outputs=output_html
    )

    gr.Markdown("""
    ---
    ### 📊 Tentang Model
    | Metrik | Nilai |
    |--------|-------|
    | Algoritma | Random Forest Regressor |
    | R² Score | ≥ 0.90 |
    | MAPE | ≤ 12% |
    | Dataset | 1.500 laptop |

    **Dikembangkan sebagai proyek tugas akhir menggunakan metodologi CRISP-DM**
    """)

if __name__ == "__main__":
    demo.launch()
