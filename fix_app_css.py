from pathlib import Path

app_path = Path("app.py")
lines = app_path.read_text().splitlines()

start = next(i for i, line in enumerate(lines) if line.startswith("def _inject_css():"))
end = next(i for i, line in enumerate(lines[start + 1 :], start + 1) if line.startswith("def _inject_js():"))

replacement = [
    "def _inject_css():",
    "    css_path = os.path.join(_HERE, \"css\", \"styles.css\")",
    "    try:",
    "        with open(css_path, \"r\") as f:",
    "            css_content = f.read()",
    "        st.markdown(f\"<style>{css_content}</style>\", unsafe_allow_html=True)",
    "    except FileNotFoundError:",
    "        st.error(f\"CSS file not found at {css_path}\")",
    "",
]

new_lines = lines[:start] + replacement + lines[end:]
app_path.write_text("\n".join(new_lines) + "\n")
print(f"Updated {app_path} from lines {start+1} to {end}.")
