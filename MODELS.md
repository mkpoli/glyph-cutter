# Models

The weights are trained, not hand-written, and carry the licence of their training data. Code in
this repository is MIT; the weights in a release are CC BY-SA 4.0.

## Files

| File | What it is |
| --- | --- |
| `detector.onnx` | RT-DETR with a ResNet-18 backbone, one class: a character box |
| `classifier.onnx`, `classes.json` | CAFormer-S18 classifier over 3,429 character classes, used only to score candidate boxes |
| `frame.json` | logistic regression: the probability that a candidate box frames one character |
| `gate.json` | logistic regression: accept a placement of a transcribed character |

`frame.json` and `gate.json` are readable JSON: standardised features, weights and a bias. They were
refitted for this tool on the Glyph Atlas cutter data without the refiner's three quality features,
because this tool ships no box refiner. The fit used 47,755 candidate boxes, 24,621 of them framing a
character, from 1,187 validation lines of CODH books and 9,253 reviewed lines. The gate accepted at
0.9655, the threshold that gave 98.5% precision on faulted validation: on CODH's four held-out test
books it places 83.8% of 44,194 characters at IoU 0.7 or better, accepts 19,361 boxes, and 97.6% of
the accepted boxes are cut right. The training scripts and the original measurements live in Glyph
Atlas (`models/detector`, `models/classifier`, `models/cutter`).

## Training data

- 『日本古典籍くずし字データセット』, ROIS-DS CODH, doi:[10.20676/00000340](https://doi.org/10.20676/00000340),
  CC BY-SA 4.0. Character boxes and crops from 44 books.
- 東京大学史料編纂所「くずし字データセット」, doi:[10.57459/hi.34.kuzushiji](https://doi.org/10.57459/hi.34.kuzushiji),
  CC BY 4.0. Character crops for the classifier.
- Reviewed crops and corrected boxes saved on Glyph Atlas, CC BY-SA 4.0, used to fit `frame.json`
  and `gate.json`.

The detector starts from `PekingU/rtdetr_r18vd` (Apache-2.0); the classifier starts from
`caformer_s18.sail_in22k_ft_in1k` through `timm` (Apache-2.0). Those checkpoints do not constrain the
licence of the fitted weights, which follows the training data above.

## Limits

The detector was trained on Japanese pre-modern printed and manuscript pages. It finds character
boxes in Chinese, Korean and Vietnamese books too, but no measurement on those traditions backs a
quality claim. It boxes characters; it does not read them. Ruby, punctuation, seals and decoration
are often not annotated, so a page may give boxes on them or miss them.

The classifier and the two logistic models were fitted on the Glyph Atlas collection. A page whose
script, layout or scan differs sharply may be cut less reliably. Check a sample before running a
whole book.
