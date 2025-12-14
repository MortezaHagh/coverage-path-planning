FROM mcr.microsoft.com/devcontainers/python:3.12

RUN apt-get update && apt-get install -y python3-opencv

# # Update package lists and install required packages
# RUN apt-get update \
#   && apt-get install -y python3-opencv \
#   && apt-get install -y --no-install-recommends ca-certificates zsh \
#   && apt-get clean \
#   && rm -rf /var/lib/apt/lists/*

# # Install LaTeX and related tools
# sudo apt update
# sudo apt install -y \
#   latexmk texlive-base texlive-latex-recommended texlive-latex-extra \
#   texlive-fonts-recommended texlive-pictures \
#   texlive-bibtex-extra biber
# # Optional extras:
# # sudo apt install -y texlive-fonts-extra python3-pygments ghostscript