/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: { sans: ["Inter", "system-ui", "sans-serif"], mono: ["JetBrains Mono", "monospace"] },
      colors: {
        ink: { DEFAULT: "#1f2328", soft: "#57606a", faint: "#8c959f" },
        line: "#e6e8eb", panel: "#f7f8fa", accent: { DEFAULT: "#4f46e5", soft: "#eef2ff" },
        pdf: "#dc2626", image: "#0891b2", audio: "#d97706", video: "#7c3aed",
      },
      boxShadow: { card: "0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.06)" },
    },
  },
  plugins: [],
};
