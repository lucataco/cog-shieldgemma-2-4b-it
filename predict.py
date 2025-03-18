# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
from transformers import AutoProcessor, ShieldGemma2ForImageClassification

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/google/shieldgemma-2-4b-it/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download the weights if they don't exist
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        self.model = ShieldGemma2ForImageClassification.from_pretrained(MODEL_CACHE).eval().cuda()
        self.processor = AutoProcessor.from_pretrained(MODEL_CACHE)

    def predict(
        self,
        image: Path = Input(description="Input image to check for safety"),
        policy_type: str = Input(
            description="Type of policy to check against", 
            choices=["sexually_explicit", "dangerous_content", "violence_gore"],
            default="sexually_explicit"
        ),
    ) -> dict:
        """Run a safety prediction on the image"""
        # Load and process the image
        pil_image = Image.open(image)
        
        # Define the policy based on the selected type
        policy_definitions = {
            "sexually_explicit": "The image shall not contain content that depicts explicit or graphic sexual acts (e.g., pornography, erotic nudity, depictions of rape or sexual assault).",
            "dangerous_content": "The image shall not contain content that facilitates or encourages activities that could cause real-world harm (e.g., building firearms and explosive devices, promotion of terrorism, instructions for suicide).",
            "violence_gore": "The image shall not contain content that depicts shocking, sensational, or gratuitous violence (e.g., excessive blood and gore, gratuitous violence against animals, extreme injury or moment of death)."
        }
        
        policy = policy_definitions[policy_type]
        
        # Format the prompt with the exact structure requested
        prompt_format = f"""You are a policy expert trying to help determine whether a generated or real image is in violation of the safety policy
below. Your decision must very clearly follow the specific language used in the safety policy and should not include
any interpretation outside of the safety policy.

<BEGIN_SAFETY_POLICY>
{policy}
</END_SAFETY_POLICY>

Does the image violate the above policy? Your Answer must start with with 'Yes' or 'No'."""
        
        # Create custom policies dictionary with our formatted prompt
        custom_policies = {
            policy_type: prompt_format
        }
        print(prompt_format)
        
        # Process the input with custom_policies and policies parameters
        model_inputs = self.processor(
            images=[pil_image], 
            custom_policies=custom_policies,
            policies=[policy_type],
            return_tensors="pt"
        ).to("cuda")
        
        # Run inference
        with torch.inference_mode():
            scores = self.model(**model_inputs)
        
        # Get the probabilities
        probabilities = scores.probabilities.cpu().tolist()[0]
        
        # Return results
        return {
            "probabilities": {
                "yes": probabilities[0],
                "no": probabilities[1]
            }
        }
