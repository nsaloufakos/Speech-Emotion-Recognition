# LaTeX preview in VS Code

Open the whole repository with **File → Open Folder** so VS Code loads the shared LaTeX settings.

## Install once

- **Linux Mint or Ubuntu** — run `sudo apt update`, then `sudo apt install latexmk texlive-latex-extra texlive-fonts-recommended`.
- **macOS** — install [MacTeX](https://tug.org/mactex/), then quit and reopen VS Code so it can discover the TeX binaries.
- **Windows** — install [MiKTeX](https://miktex.org/download) and [Strawberry Perl](https://strawberryperl.com/) for `latexmk`. Enable automatic package installation in MiKTeX Console and restart VS Code.

Install [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) from the VS Code Extensions panel, or run `code --install-extension James-Yu.latex-workshop` in a terminal that has the `code` command.

## Edit with the PDF on the right

1. Open `FINAL CONTENT/final-template.tex` as a setup test, or open the main `.tex` file for the report or slides.
2. Open the Command Palette and run **LaTeX Workshop: Build LaTeX project**.
3. Run **LaTeX Workshop: View LaTeX PDF file**, or click the PDF preview icon in the editor toolbar. The source stays on the left and the PDF opens on the right. The default shortcut is **Ctrl+Alt+V** on Linux and Windows or **Cmd+Option+V** on macOS.
4. Save the source to rebuild and refresh the PDF. Each document's generated PDF and auxiliary files are in the ignored `.latex-build` folder next to its source.

If the compiler is missing, check `latexmk -v` and `pdflatex --version` in a new VS Code terminal. If both work and a build still fails, open **View → Output** and select **LaTeX Compiler** for the error and source line.

You can also build from the repository root in a terminal:

```sh
latexmk -cd -pdf -synctex=1 -interaction=nonstopmode -halt-on-error -file-line-error -outdir=.latex-build "FINAL CONTENT/final-template.tex"
```
