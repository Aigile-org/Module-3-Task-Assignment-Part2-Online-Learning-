
class TransformUnion_calc:
    """
    A custom implementation of an online transformer union.
    
    This class takes multiple transformer objects and applies them in parallel.
    It combines their resulting feature dictionaries into a single, larger one,
    ensuring feature keys are unique by prefixing them with the transformer's name.
    """
    def __init__(self, *transformers):
        """
        Args:
            *transformers: A variable number of transformer objects. Each object
                         is expected to have `learn_one`, `transform_one` methods,
                         and an `on` attribute to identify its input field.
        """
        self.transformers = transformers

    def learn_one(self, x: dict):
        """Calls `learn_one` on each internal transformer."""
        for transformer in self.transformers:
            transformer.learn_one(x)
        return self

    def transform_one(self, x: dict):
        """
        Calls `transform_one` on each transformer and merges the results.
        
        Feature keys are made unique by prefixing them with the name of the
        field the transformer operated on (e.g., 'summary_0', 'description_15').
        """
        combined_features = {}
        for transformer in self.transformers:
            # Get the sparse feature vector from the individual transformer
            individual_features = transformer.transform_one(x)
            
            # Get the name of the feature field to use as a prefix
            prefix = transformer.on
            
            # Merge the features, creating a new, unique key for each
            for key, value in individual_features.items():
                new_key = f"{prefix}_{key}"
                combined_features[new_key] = value
                
        return combined_features
