# ShieldGemma-2-4b-it

This is a Replicate Cog implementation of Google's [ShieldGemma 2](https://ai.google.dev/gemma/docs/shieldgemma) model for image safety classification. The model takes an input image and classifies whether it violates specific safety policies.

## Model Description

ShieldGemma 2 is a model trained on Gemma 3's 4B IT checkpoint for image safety classification across key categories. It helps determine if images contain harmful content across three key areas:

- **Sexually Explicit Content**: Content depicting explicit or graphic sexual acts
- **Dangerous Content**: Content facilitating activities that could cause real-world harm
- **Violence/Gore Content**: Content depicting shocking, sensational, or gratuitous violence

## Usage

### Local Development

To run the model locally:

1. Install [Cog](https://github.com/replicate/cog):
```bash
curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
chmod +x /usr/local/bin/cog
```

2. Download the model weights:
```bash
python download.py
```

3. Run the model:
```bash
# Example with a bee image
cog predict -i image=@bee.jpg
```

4. You can specify the policy type:
```bash
cog predict -i image=@bee.jpg -i policy_type=violence_gore
```

### Deploying to Replicate

To deploy the model to Replicate:

1. Push the model to Replicate:
```bash
cog push r8.im/username/shieldgemma-2-4b-it
```

2. Once deployed, you can use it via the Replicate API:
```python
import replicate

output = replicate.run(
    "username/shieldgemma-2-4b-it:latest",
    input={
        "image": open("bee.jpg", "rb"),
        "policy_type": "sexually_explicit"
    }
)
print(output)
```

### Sample Output

When running with a sample bee image:

```bash
cog predict -i image=@bee.jpg
```

You might get an output like:

```json
{
  "policy_type": "sexually_explicit",
  "policy_definition": "The image shall not contain content that depicts explicit or graphic sexual acts (e.g., pornography, erotic nudity, depictions of rape or sexual assault).",
  "violates_policy": false,
  "confidence": 0.98,
  "probabilities": {
    "no": 0.98,
    "yes": 0.02
  }
}
```

## API Reference

### Input Parameters

- `image`: The input image to check for safety violations (required)
- `policy_type`: The type of policy to check against (optional, default: "sexually_explicit")
  - Options: "sexually_explicit", "dangerous_content", "violence_gore"

### Output

The model returns a JSON object with:

- `policy_type`: The type of policy checked
- `policy_definition`: The full definition of the policy
- `violates_policy`: Boolean indicating if the image violates the policy
- `confidence`: The confidence score of the prediction
- `probabilities`: Object containing the "yes" and "no" probabilities

## Limitations

- The model is based on Gemma 3 and shares its general limitations
- Performance may vary based on image quality and content
- The model is highly sensitive to the specific user-provided description of safety principles
- Limited benchmarks for content moderation mean the training and evaluation data might not be representative of all real-world scenarios

## License

This model is subject to Google's [Gemma license terms](https://ai.google.dev/gemma/terms).

## Citation

```
@article{shieldgemma2,
    title={ShieldGemma 2},
    url={https://ai.google.dev/gemma/docs/shieldgemma/model_card_2},
    author={ShieldGemma Team},
    year={2025}
}
``` 