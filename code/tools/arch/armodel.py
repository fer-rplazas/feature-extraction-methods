import torch
import torch.nn as nn

from .cnn1d import CNN1d


class ARModel(nn.Module):
    def __init__(self, n_channels: int, n_feats: int, cnn_hparams: dict):

        super().__init__()

        self.convs = CNN1d(n_channels=n_channels, n_out=45, **cnn_hparams)
        self.feat_encoder = nn.Sequential(
            nn.BatchNorm1d(n_feats), nn.Linear(n_feats, 15)
        )
        self.combiner = nn.Sequential(
            nn.Linear(45 + 15, 10), nn.BatchNorm1d(10), nn.SiLU()
        )

        self.AR = nn.LSTM(10, 2, proj_size=1)

    def forward(self, signals: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """

        Input:
        ------
            signals (torch.Tensor): Sequenced and batched multichannel timeseries signals. Shape (n_seq, n_batch, n_chan, n_times)
            feats (torch.Tensor): Sequenced and batched scalar features. Shape (n_seq, n_batch, n_feats)

        Returns:
        --------
            out (torch.Tensor): Logits. Shape (n_seq, n_batch)
        """

        # l_ = []
        # for x in torch.unbind(signals, 0):
        #     l_.append(self.convs(x))
        # conved = torch.stack(l_)  # (n_seq, n_batch, n_conv_feats)

        conved = torch.stack(
            [self.convs(x) for x in torch.unbind(signals, 0)]
        )  # (n_seq, n_batch, n_conv_feats)

        # l_ = []
        # for x in torch.unbind(feats, 0):
        #     l_.append(self.feat_encoder(x))
        # features_encoded = torch.stack(l_)  # (n_seq, n_batch, n_feats_encoded)

        features_encoded = torch.stack(
            [self.feat_encoder(x) for x in torch.unbind(feats, 0)]
        )  # (n_seq, n_batch, n_feats_encoded)

        feats_both = torch.cat((conved, features_encoded), -1)

        # l_ = []
        # for x in torch.unbind(feats_both, 0):
        #     l_.append(self.combiner(x))
        # feats_all = torch.stack(l_)  # (n_seq, n_batch, n_feats_combined)

        feats_all = torch.stack(
            [self.combiner(x) for x in torch.unbind(feats_both, 0)]
        )  # (n_seq, n_batch, n_feats_combined)

        out, _ = self.AR(feats_all)  # out (n_seq, n_batch, 1)
        out = out.squeeze()

        return out
