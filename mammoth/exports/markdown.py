from kfp import dsl


class Markdown:
    integration = "dsl.Markdown"

    def __init__(self, text):
        self._text = text

    def show(self, title="MAMMOth-commons Markdown"):
        import markdown2
        from mammoth.exports.HTML import HTML

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
        </head>
        <body>
            {markdown2.markdown(self._text, extras=["tables", "fenced-code-blocks", "code-friendly"])}
        </body>
        </html>
        """
        HTML(html).show()

    def text(self):
        import markdown2
        from mammoth.exports.HTML import HTML

        return HTML(
            markdown2.markdown(
                self._text, extras=["tables", "fenced-code-blocks", "code-friendly"]
            )
        ).text()

    def export(self, output: dsl.Output[integration]):
        with open(output.path, "w") as f:
            output.name = "result.md"
            f.write(self._text)
