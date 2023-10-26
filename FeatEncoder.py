import torch.nn as nn
class VisualFeatEncoder(nn.Module):
    def __init__(self, visual_feat_dim,visual_pos_dim,hidden_size):
        super().__init__()
        feat_dim = visual_feat_dim
        pos_dim = visual_pos_dim

        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, hidden_size)
        self.visn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        # Box position encoding
        self.box_fc = nn.Linear(pos_dim, hidden_size)
        self.box_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(0.1)

    def forward(self, feats,boxes):

        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        y = self.box_fc(boxes)
        y = self.box_layer_norm(y)
        output = (x + y) / 2

        output = self.dropout(output)
        return output