import h5py
import json
import numpy as np
import torch
import os
from PIL import Image

class TextFeatureGenerator:
    def __init__(self, model_manager, features_json_path, output_path="text_features.h5"):
        """Initialize text feature generator with detailed logging"""
        print(f"\nInitializing TextFeatureGenerator:")
        print(f"- Features JSON path: {features_json_path}")
        print(f"- Output path: {output_path}")
        
        self.model_manager = model_manager
        self.features_json_path = features_json_path
        self.output_path = output_path
        
        # Initialize AIMv2 components
        print("Loading AIMv2 model and processor...")
        self.model = self.model_manager.get_aimv2_model()
        self.processor = self.model_manager.get_aimv2_processor()
        print("Model and processor loaded successfully")
        
        # Create a default blank image for text processing
        self.default_image = Image.new('RGB', (224, 224), color='white')

    def generate_descriptions(self, features_data):
        """Generate text descriptions with logging"""
        print("\nGenerating descriptions from features data:")
        descriptions = {}
        total_descriptions = 0
        
        for category, attributes in features_data.items():
            category_descriptions = []
            
            if isinstance(attributes, dict):
                # Handle nested attributes
                if "attributes" in attributes:
                    for attr_type, values in attributes["attributes"].items():
                        for value in values:
                            desc = f"a person with {value} {attr_type}"
                            category_descriptions.append(desc)
                            
                # Handle direct attributes
                for key, values in attributes.items():
                    if isinstance(values, list) and key not in ["views", "attributes"]:
                        for value in values:
                            desc = f"a {value} {category.lower()}"
                            category_descriptions.append(desc)
                            
                # Add combinations
                if category in ["Hair", "Upper-clothes", "Dress"]:
                    if "colors" in attributes and "styles" in attributes:
                        for color in attributes["colors"]:
                            for style in attributes["styles"]:
                                desc = f"a {color} {style} {category.lower()}"
                                category_descriptions.append(desc)
                                
            if category_descriptions:
                descriptions[category] = category_descriptions
                total_descriptions += len(category_descriptions)
                print(f"Generated {len(category_descriptions)} descriptions for {category}")
                
        print(f"\nTotal descriptions generated: {total_descriptions}")
        return descriptions

    def process_text_features(self, descriptions):
        """Process text descriptions with proper image input"""
        try:
            print("\nProcessing text features:")
            features = {}
            
            for category, category_descriptions in descriptions.items():
                print(f"\nProcessing category: {category}")
                print(f"Number of descriptions: {len(category_descriptions)}")
                
                batch_size = 32
                all_features = []
                
                # Process in batches
                for i in range(0, len(category_descriptions), batch_size):
                    batch = category_descriptions[i:i + batch_size]
                    batch_end = min(i + batch_size, len(category_descriptions))
                    print(f"  Processing batch {i//batch_size + 1}: items {i} to {batch_end}")
                    
                    # Prepare inputs with both image and text
                    inputs = self.processor(
                        images=[self.default_image] * len(batch),  # Duplicate default image for batch
                        text=batch,
                        add_special_tokens=True,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(self.model.device)
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        # Get text features and normalize
                        text_features = outputs.text_features.cpu().numpy()
                        # Normalize features
                        norms = np.linalg.norm(text_features, axis=1, keepdims=True)
                        text_features = text_features / norms
                        all_features.append(text_features)
                        
                    print(f"    Processed {len(batch)} descriptions")
                
                # Combine batches
                category_features = np.vstack(all_features)
                print(f"Features shape for {category}: {category_features.shape}")
                
                features[category] = {
                    'features': category_features,
                    'descriptions': category_descriptions
                }
            
            return features
            
        except Exception as e:
            print(f"Error processing text features: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_features(self, features):
        """Save features with detailed error checking"""
        try:
            print(f"\nSaving features to {self.output_path}")
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(self.output_path) if os.path.dirname(self.output_path) else '.', exist_ok=True)
            
            # Save to HDF5
            with h5py.File(self.output_path, 'w') as f:
                for category, data in features.items():
                    print(f"\nSaving category: {category}")
                    
                    # Create group
                    category_group = f.create_group(category)
                    
                    # Save features
                    print(f"Saving features array: {data['features'].shape}")
                    category_group.create_dataset('features', data=data['features'])
                    
                    # Save descriptions
                    desc_json = json.dumps(data['descriptions'])
                    print(f"Saving descriptions: {len(data['descriptions'])} items")
                    category_group.create_dataset('descriptions', data=desc_json.encode())
                    
            print(f"\nSuccessfully saved features to {self.output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving features: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_and_save_features(self):
        """Main process with error handling"""
        try:
            print(f"\nStarting feature generation process")
            
            # Load and parse JSON
            with open(self.features_json_path, 'r', encoding='utf-8') as f:
                features_data = json.load(f)
            print("Successfully loaded features.json")
            
            # Generate descriptions
            print("\nGenerating descriptions...")
            descriptions = self.generate_descriptions(features_data)
            if not descriptions:
                print("Error: No descriptions generated")
                return False
                
            # Process features
            print("\nProcessing features...")
            features = self.process_text_features(descriptions)
            if not features:
                print("Error: Feature processing failed")
                return False
            
            # Save results
            print("\nSaving features...")
            return self.save_features(features)
            
        except Exception as e:
            print(f"\nError in generate_and_save_features: {e}")
            import traceback
            traceback.print_exc()
            return False

    @classmethod
    def load_text_features(cls, feature_path):
        """Load features with validation"""
        try:
            print(f"\nLoading text features from: {feature_path}")
            
            if not os.path.exists(feature_path):
                print(f"Error: Feature file not found: {feature_path}")
                return None
            
            text_features = {}
            with h5py.File(feature_path, 'r') as f:
                for category in f.keys():
                    print(f"Loading category: {category}")
                    features = f[category]['features'][:]
                    descriptions = json.loads(f[category]['descriptions'][()])
                    print(f"- Loaded {len(descriptions)} descriptions")
                    print(f"- Features shape: {features.shape}")
                    
                    text_features[category] = {
                        'features': features,
                        'descriptions': descriptions
                    }
            
            print("Successfully loaded all features")
            return text_features
            
        except Exception as e:
            print(f"Error loading text features: {e}")
            import traceback
            traceback.print_exc()
            return None