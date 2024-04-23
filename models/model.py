from einops import rearrange

from models.diffusion.utils import nonlinearity, get_timestep_embedding
from models.backbone.ChebConv import _GraphConv
from models.backbone.GraFormer import *


class _ResChebGC_diff(nn.Module):
    def __init__(self, adj, input_dim, output_dim, emd_dim, hid_dim, p_dropout):
        super(_ResChebGC_diff, self).__init__()
        self.adj = adj
        self.gconv1 = _GraphConv(input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(hid_dim, output_dim, p_dropout)
        # time embedding
        self.temb_proj = torch.nn.Linear(emd_dim, hid_dim)

    def forward(self, x, temb):
        residual = x
        out = self.gconv1(x, self.adj)
        out = out + self.temb_proj(nonlinearity(temb))[:, None, :]
        out = self.gconv2(out, self.adj)
        return residual + out


class DiffModel(nn.Module):
    def __init__(self, adj_2d3d, adj_ctx,
                 hid_dim, coords_dim, num_layers,
                 n_head, dropout, n_pts):
        super(DiffModel, self).__init__()

        # skeleton ############################
        self.adj_2d3d = adj_2d3d
        # load gcn configuration
        self.hid_dim = hid_dim
        self.emd_dim = hid_dim
        self.coords_dim = coords_dim
        self.hid_dim = self.hid_dim
        self.emd_dim = self.hid_dim * 4
        self.n_layers = num_layers

        # 2d-3d cross-channel feature
        _gconv_input = ChebConv(in_c=self.coords_dim[0], out_c=self.hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []
        dim_model = self.hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, dim_model)
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=n_pts)
        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC_diff(adj=adj_2d3d, input_dim=self.hid_dim, output_dim=self.hid_dim,
                                                 emd_dim=self.emd_dim, hid_dim=self.hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))
        self.gconv_input = _gconv_input
        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)
        self.gconv_output = ChebConv(in_c=dim_model, out_c=self.coords_dim[1], K=2)

        # 2d context guidance
        self.adj_ctx = adj_ctx
        _gconv_input_t = ChebConv(in_c=2, out_c=self.hid_dim, K=2)
        _gconv_layers_t = []
        _attention_layer_t = []
        gcn_t = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=34)
        for i in range(num_layers):
            _gconv_layers_t.append(_ResChebGC_diff(adj=adj_2d3d, input_dim=self.hid_dim, output_dim=self.hid_dim,
                                                   emd_dim=self.emd_dim, hid_dim=self.hid_dim, p_dropout=0.1))
            _attention_layer_t.append(GraAttenLayer(dim_model, c(attn), c(gcn_t), dropout))
        self.gconv_input_t = _gconv_input_t
        self.gconv_layers_t = nn.ModuleList(_gconv_layers_t)
        self.atten_layers_t = nn.ModuleList(_attention_layer_t)
        self.gconv_output_t = ChebConv(in_c=dim_model, out_c=self.coords_dim[1], K=2)

        # dimensional alignment
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(6, 5)
        self.fc3 = nn.Linear(17, 34)

        # diffusion configuration
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.hid_dim, self.emd_dim),
            torch.nn.Linear(self.emd_dim, self.emd_dim),
        ])

    def forward(self, x, guide, mask, t):
        b, f, j, _ = x.shape

        # time step embedding
        temb = get_timestep_embedding(t, self.hid_dim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # cross-channel 2D-3D feature
        x_s = rearrange(x, 'b f j d -> (b f) j d')
        x2d, x3d = x_s.split([2, 3], dim=-1)
        x2d, x3d = x2d.float(), x3d.float()
        x2d = self.fc1(x2d)
        x_2d3d = torch.cat((x2d, x3d), dim=-2)
        mask = mask.repeat(1, 1, 2)

        # 2D context guidance
        guide = rearrange(guide, 'b f j d -> (b f) j d')
        guide = self.gconv_input_t(guide, self.adj_ctx)
        guide = rearrange(self.fc3(rearrange(guide, 'b j d -> b d j')), 'b d j -> b j d')

        # two branches inference
        out = self.gconv_input(x_2d3d, self.adj_2d3d) + guide
        for i in range(self.n_layers - 1):
            out = self.atten_layers[i](out, mask)
            guide = self.atten_layers_t[i](guide, mask)

            # feature merging
            out += guide
            guide += out

            out = self.gconv_layers[i](out, temb)
            guide = self.gconv_layers_t[i](guide, temb)

        # last layer
        out = self.atten_layers[-1](out, mask)
        guide = self.atten_layers_t[-1](guide, mask)
        out += guide
        out = self.gconv_layers[-1](out, temb)
        out = self.gconv_output(out, self.adj_2d3d)

        # read out signal
        out_2d, out_3d = out.split([17, 17], dim=-2)
        out_2d, out_3d = out_2d.float(), out_3d.float()
        out = torch.cat((out_2d, out_3d), dim=-1)
        out = self.fc2(out)
        out = rearrange(out, '(b f) j d -> b f j d', b=b, j=j, f=f)
        return out
