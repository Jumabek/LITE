import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


class Evaluator:
    @staticmethod
    def __call__(features):
        """Calculates positive and negative similarities between features."""
        pos_matches, neg_matches = [], []
        features['features'] = features['features'].apply(lambda x: x / np.linalg.norm(x))
        num_frames = len(features['frame'].unique())

        for i in tqdm(range(1, num_frames), desc="Calculating similarity"):
            current_frame_data = features[features['frame'] == i]
            track_ids = current_frame_data['id'].unique()

            for track_id in track_ids:
                feat = current_frame_data[current_frame_data['id']
                                          == track_id].features.values[0].reshape(1, -1)
                feat_pos = features[(features['id'] == track_id) & (
                    features['frame'] > i)].features.values
                #  & (features['frame'] < i+30) # if max_age added

                feat_neg = current_frame_data[current_frame_data['id']
                                              != track_id].features # [:feat_pos.shape[0]] # if max_age added
                 
                if feat_pos.size > 0:
                    pos_sim = cosine_similarity(feat, np.vstack(
                        [f.reshape(1, -1) for f in feat_pos]))
                    pos_matches.extend(pos_sim)

                if feat_neg.size > 0:
                    neg_sim = cosine_similarity(feat, np.vstack(
                        [f.reshape(1, -1) for f in feat_neg]))
                    neg_matches.extend(neg_sim)

        pos_matches = np.concatenate(pos_matches)
        neg_matches = np.concatenate(neg_matches)
        
        # randomly sample negative matches to match the number of positive matches
        if len(neg_matches) > len(pos_matches):
            neg_matches = np.random.choice(neg_matches, len(pos_matches), replace=False)

        elif len(neg_matches) < len(pos_matches):
            pos_matches = np.random.choice(pos_matches, len(neg_matches), replace=False)

        return pos_matches, neg_matches
