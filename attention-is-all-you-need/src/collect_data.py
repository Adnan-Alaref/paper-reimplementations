from src.imports import *
  
# ── Preprocess English and Arabic texts ────────────────────────────────────────────────────────
def preprocess_languages(
  eng_data_path: str,
  arb_data_path: str
  )->tuple[List[str], List[str]]:

  """
    Read aligned English and Arabic sentences from text files.
    Each line in both files represents one translation pair.
  """

  with open(file=eng_data_path, mode='r', encoding='utf-8') as eng_file:
    eng_data = [line.strip() for line in eng_file]

  with open(file=arb_data_path, mode='r', encoding='utf-8') as arb_file:
    arb_data = [line.strip() for line in arb_file]

  if len(eng_data) != len(arb_data):
    raise ValueError(
      "English and Arabic files must contain the same number "
      "of sentences."
    )

  return eng_data, arb_data