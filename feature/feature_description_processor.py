import json
from typing import List, Dict, Any, Optional
import random

class FeatureDescriptionProcessor:
    def __init__(self, features_json_path: str):
        with open(features_json_path, 'r') as f:
            self.features_data = json.load(f)
            
    def process_features(self, image) -> Dict[str, List[str]]:
        """Process image and return top matches for each feature category"""
        feature_matches = {
            "appearance": self._get_appearance_matches(),
            "clothing": self._get_clothing_matches(),
            "behavior": self._get_behavior_matches(),
            "products": self._get_product_matches()
        }
        return feature_matches
        
    def _get_appearance_matches(self, top_k: int = 5) -> List[str]:
        """Get top appearance feature matches"""
        matches = []
        customer = self.features_data["Customer"]["Appearance"]
        
        for _ in range(min(top_k, 5)):
            desc = f"a {random.choice(customer['Height'])} {random.choice(customer['Build'])} "
            desc += f"{random.choice(customer['AgeGroup'])} {random.choice(customer['Gender'])}"
            matches.append(desc)
            
        return matches
        
    def _get_clothing_matches(self, top_k: int = 5) -> List[str]:
        """Get top clothing feature matches"""
        matches = []
        clothing = self.features_data["Customer"]["Clothing"]
        
        for _ in range(min(top_k, 5)):
            upper = clothing["UpperBody"]
            lower = clothing["LowerBody"]
            
            desc = f"wearing {random.choice(upper['Colors'])} {random.choice(upper['Types'])} "
            desc += f"and {random.choice(lower['Colors'])} {random.choice(lower['Types'])}"
            matches.append(desc)
            
        return matches
        
    def _get_behavior_matches(self, top_k: int = 5) -> List[str]:
        """Get top behavior feature matches"""
        matches = []
        behavior = self.features_data["Shopping_Behavior"]
        
        for _ in range(min(top_k, 5)):
            action = random.choice(behavior["Actions"]["Movement"])
            interaction = random.choice(behavior["Actions"]["Interaction"])
            desc = f"{action} while {interaction}"
            matches.append(desc)
            
        return matches
        
    def _get_product_matches(self, top_k: int = 5) -> List[str]:
        """Get top product interaction matches"""
        matches = []
        products = self.features_data["Products"]["Categories"]
        
        for _ in range(min(top_k, 5)):
            category = random.choice(list(products.keys()))
            product = random.choice(products[category]["Types"])
            
            if "Flavors" in products[category]:
                flavor = random.choice(products[category]["Flavors"])
                desc = f"looking at {flavor} {product} in {category}"
            else:
                desc = f"examining {product} in {category}"
            matches.append(desc)
            
        return matches

    def generate_combined_description(self, features: Dict[str, List[str]]) -> str:
        """Generate a complete description from matched features"""
        description = []
        
        if features.get("appearance"):
            description.append(random.choice(features["appearance"]))
        if features.get("clothing"):
            description.append(random.choice(features["clothing"]))
        if features.get("behavior"):
            description.append(random.choice(features["behavior"]))
        if features.get("products"):
            description.append(random.choice(features["products"]))
            
        return " ".join(description)

    def get_similarity_scores(self, query_features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate similarity scores for each feature category"""
        scores = {}
        
        if "appearance" in query_features:
            matches = self._get_appearance_matches()
            # Implement similarity calculation here
            scores["appearance"] = random.uniform(0.5, 1.0)
            
        if "clothing" in query_features:
            matches = self._get_clothing_matches()
            scores["clothing"] = random.uniform(0.5, 1.0)
            
        if "behavior" in query_features:
            matches = self._get_behavior_matches()
            scores["behavior"] = random.uniform(0.5, 1.0)
            
        if "products" in query_features:
            matches = self._get_product_matches()
            scores["products"] = random.uniform(0.5, 1.0)
            
        return scores

if __name__ == "__main__":
    processor = FeatureDescriptionProcessor("features.json")
    
    # Get feature matches
    features = processor.process_features(None)
    
    print("\nFeature matches by category:")
    for category, matches in features.items():
        print(f"\n{category.title()}:")
        for match in matches:
            print(f"- {match}")
            
    print("\nCombined description:")
    print(processor.generate_combined_description(features))