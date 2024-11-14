import pytest

from keras_hub.src.models.clip.clip_preprocessor import CLIPPreprocessor
from keras_hub.src.models.clip.clip_tokenizer import CLIPTokenizer
from keras_hub.src.models.flux.flux_text_to_image_preprocessor import (
    FluxTextToImagePreprocessor,
)
from keras_hub.src.tests.test_case import TestCase


class FluxTextToImagePreprocessorTest(TestCase):
    def setUp(self):
        vocab = ["air", "plane</w>", "port</w>"]
        vocab += ["<|endoftext|>", "<|startoftext|>"]
        vocab = dict([(token, i) for i, token in enumerate(vocab)])
        merges = ["a i", "p l", "n e</w>", "p o", "r t</w>", "ai r", "pl a"]
        merges += ["po rt</w>", "pla ne</w>"]
        clip_l_tokenizer = CLIPTokenizer(
            vocabulary=vocab, merges=merges, pad_with_end_token=True
        )
        clip_l_preprocessor = CLIPPreprocessor(
            clip_l_tokenizer, sequence_length=8
        )
        self.init_kwargs = {
            "clip_l_preprocessor": clip_l_preprocessor,
        }
        self.input_data = ["airplane"]

    def test_preprocessor_basics(self):
        pytest.skip(
            reason="TODO: enable after preprocessor flow is figured out"
        )
        self.run_preprocessing_layer_test(
            cls=FluxTextToImagePreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[1, 4, 9, 5, 7, 2, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
                },
                [[4, 9, 5, 7, 2, 0, 0, 0]],  # Labels shifted.
                [[1, 1, 1, 1, 1, 0, 0, 0]],  # Zero out unlabeled examples.
            ),
        )

    def test_generate_preprocess(self):
        preprocessor = FluxTextToImagePreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(self.input_data)
        self.assertIn("clip_l", x)
        self.assertAllEqual(x["clip_l"][0], [4, 0, 1, 3, 3, 3, 3, 3])