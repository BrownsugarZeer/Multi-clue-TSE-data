## Readme

This is the data simulation scirpt for paper "Target Sound Extraction (TSE) with Variable Cross-modality Clues".

### How to use:

1. Clone this project: `git clone --recursive https://github.com/LiChenda/Multi-clue-TSE-data.git`
2. Install pytorch: `pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111`
3. Install requirements: `pip install -r requirements.txt`
4. Download the AudioSet and AudioCaps dataset. 
5. Run simulation script: `python data_simulation.py`
6. Prepare tag clues: `python gen_tag_clue.py`, the one-hot tag will be created in `output/[train|val|test|unseen]/tag_onehot/`.
7. Prepare text clues: `python gen_text_clue.py` .
8. Prepare visual clues: `python gen_visual_clue.py` .

### Supported clues:

- [x] Tag clue
- [x] Video clue
- [x] Text clue

