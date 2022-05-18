import numpy as np
import os, glob
import torch
from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsClassifier


class EmbeddingsHandler:
    def __init__(self, dataset_dir, threshold=0.4, n_neighbors=30):
        self.root_dir = dataset_dir
        self.mean_embedding = {}
        self.data_dict = {}
        self.name_dict = {}
        self.n_neighbors = n_neighbors
        self._load_dataset()
        self.threshold = threshold

        self.excluded_entities = []
        self.similarity_func = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def _load_dataset(self):
        """
        Load the set of embeddings define in the root_dir
        :return:
        """
        speaker_labels = os.listdir(self.root_dir)
        for label_id, s in enumerate(speaker_labels):
            emb_filenames = glob.glob(os.path.join(self.root_dir, s+'/*', "*.npy"))
            list_emb = [np.load(emb_f).squeeze() for emb_f in emb_filenames]

            mean = np.array(list_emb).mean(axis=0)
            self.mean_embedding[s] = mean
            self.data_dict[s] = list_emb
            self.name_dict[label_id] = s
            # print(self.data_dict)

    def get_distance_from_user(self, emb, identity_to_check):
        max_dist = -1

        if identity_to_check in self.data_dict.keys():
            for person_emb in self.data_dict[identity_to_check]:
                dist = self.similarity_func(torch.from_numpy(person_emb), emb).numpy()
                if dist[0] > max_dist:
                    max_dist = dist[0]

            return max_dist
        return False

    def get_max_distances(self, emb, thr=None):
        if thr is None:
            thr = self.threshold
        list_distance = []
        label_list = []
        print("Data dictionary size {}".format(len(self.data_dict)))
        for speaker_label, list_emb in self.data_dict.items():
            if speaker_label not in self.excluded_entities:
                for person_emb in list_emb:
                    # dist = self.similarity_func(torch.from_numpy(person_emb), torch.from_numpy(np.ndarray(np.int(emb)))).numpy()
                    dist = self.similarity_func(torch.from_numpy(person_emb), emb).numpy()
                    if dist > thr:
                        list_distance.append(dist[0])
                        label_list.append(speaker_label)

        return list_distance, label_list

    def get_speaker_db_scan(self, emb, thr=None):
        distances, labels = self.get_max_distances(emb, thr)
        if len(distances) == 0:
            return -1, -1

        n = len(distances) if len(distances) < self.n_neighbors else self.n_neighbors
        try:
            max_dist_idx = np.argpartition(distances, -n)[-n:]
            count = dict()
            for i in max_dist_idx:
                if labels[i] not in count.keys():
                    count[labels[i]] = [0, distances[i]]
                    continue
                count[labels[i]][0] += 1
                count[labels[i]][1] += distances[i]

            max_count = 0
            max_dist = 0
            final_label = ''
            for key, value in count.items():
                if value[0] >= max_count and value[1] >= max_dist:
                    max_count = value[0]
                    max_dist = value[1]
                    final_label = key

            self.excluded_entities.append(final_label)
            return max_dist / max_count, final_label

        except Exception as e:
            return -1, -1

    def get_speaker(self, emb):
        min_score = 0
        final_label = 0
        for speaker_label, mean_emb in self.mean_embedding.items():
            score = self.similarity_func(torch.from_numpy(mean_emb), emb)
            score = score.mean()
            if score > min_score:
                min_score = score
                final_label = speaker_label

        min_score = 0
        for embeddings in self.data_dict[final_label]:
            score = self.similarity_func(torch.from_numpy(embeddings), emb)
            score = score.mean()

            if score > min_score:
                min_score = score

        return min_score, final_label


if __name__ == '__main__':
    OUTPUT_EMB_TRAIN = "/home/PycharmProjects/SoundAnalyzer/data/dataset_emb/train"

    speaker_emb = EmbeddingsHandler(OUTPUT_EMB_TRAIN)

    print(speaker_emb.name_dict)
