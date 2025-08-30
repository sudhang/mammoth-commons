from kfp import dsl
import base64
import re


def _encode_image_to_base64(filepath):
    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def _replace_emojis(text):
    emoji_patterns = {
        ":x:": "❌",
        ":rocket:": "🚀",
        ":checkmark:": "🗸",
        ":smile:": "😄",
        ":thumbsup:": "👍",
        ":heart:": "❤️",
        ":star:": "⭐",
        ":fire:": "🔥",
        ":tada:": "🎉",
        ":clap:": "👏",
        ":heavy_check_mark:": "✔️",
    }
    pattern = re.compile("|".join(re.escape(key) for key in emoji_patterns.keys()))
    return pattern.sub(lambda m: emoji_patterns[m.group(0)], text)


def _highlight_code(html_content):
    try:
        from pygments import highlight
        from pygments.lexers import get_lexer_by_name
        from pygments.formatters import HtmlFormatter

        code_block_pattern = re.compile(
            r'<pre><code class="language-(\w+)">(.*?)</code></pre>', re.DOTALL
        )

        def replace_code_block(match):
            lang = match.group(1)
            code = (
                match.group(2)
                .replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&amp;", "&")
            )
            lexer = get_lexer_by_name(lang)
            formatter = HtmlFormatter()
            return highlight(code, lexer, formatter)

        return code_block_pattern.sub(replace_code_block, html_content)
    except:
        print("Consider install pygments (pip install pygments) to highlight HTML code")
        return html_content


class HTML:
    integration = "dsl.HTML"

    def __init__(self, body, header="", script="", images=None):
        self.body = body
        self.header = header
        self.script = script
        self.images = dict() if images is None else images

    def show(self):
        import webbrowser

        with open("temp.html", "w", encoding="utf-8") as file:
            file.write(self.text())
        webbrowser.open_new(file.name)

    def text(self):
        body = self.body
        for image, path in self.images.items():
            data = _encode_image_to_base64(path)
            img = f'<img src="base64,{data}" alt="{image}" />'
            body.replace(image, img)
        return _highlight_code(_replace_emojis(body))

    def export(self, output: dsl.Output[integration]):
        with open(output.path, "w") as f:
            output.name = "result.html"
            f.write(self.text())
