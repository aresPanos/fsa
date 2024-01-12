import torch
import torch.nn as nn
from cnn.efficientnet import film_efficientnet


def create_feature_extractor():
    feature_extractor = film_efficientnet()

    # freeze the parameters of feature extractor
    for param in feature_extractor.parameters():
        param.requires_grad = False

    return feature_extractor


def create_film_adapter(feature_extractor, task_dim: int=65):
    adaptation_config = feature_extractor.get_adaptation_config()
    feature_adapter = FilmAdapter(layer=FilmLayer, adaptation_config=adaptation_config, task_dim=task_dim)

    return feature_adapter


class BaseFilmLayer(nn.Module):
    def __init__(self, num_maps, num_blocks):
        super(BaseFilmLayer, self).__init__()

        self.num_maps = num_maps
        self.num_blocks = num_blocks
        self.num_generated_params = 0


    def regularization_term(self):
        """
        Compute the regularization term for the parameters. Recall, FiLM applies gamma * x + beta. As such, params
        gamma and beta are regularized to unity, i.e. ||gamma - 1||_2 and ||beta||_2.
        :return: (torch.tensor) Scalar for l2 norm for all parameters according to regularization scheme.
        """
        l2_term = 0
        for gamma_regularizer, beta_regularizer in zip(self.gamma_regularizers, self.beta_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        return l2_term


class FilmLayer(BaseFilmLayer):
    def __init__(self, num_maps, num_blocks, task_dim=None, init_values=None):
        BaseFilmLayer.__init__(self, num_maps, num_blocks)

        self.gammas = nn.ParameterList()
        self.betas = nn.ParameterList()
        
        '''
        if self.num_maps[0] < 1400:
            for i in range(self.num_blocks):
                self.gammas.append(nn.Parameter(torch.ones(self.num_maps[i]), requires_grad=True))
                self.betas.append(nn.Parameter(torch.zeros(self.num_maps[i]), requires_grad=True))

        else:
            for i in range(self.num_blocks):
                self.gammas.append(nn.Parameter(torch.ones(self.num_maps[i]), requires_grad=False))
                self.betas.append(nn.Parameter(torch.zeros(self.num_maps[i]), requires_grad=False))

        '''

        #'''
        if init_values is None:
            for i in range(self.num_blocks):
                self.gammas.append(nn.Parameter(torch.ones(self.num_maps[i]), requires_grad=True))
                self.betas.append(nn.Parameter(torch.zeros(self.num_maps[i]), requires_grad=True))
        else:
            for i in range(self.num_blocks):
                self.gammas.append(nn.Parameter(init_values[i]['gamma'], requires_grad=True))
                self.betas.append(nn.Parameter(init_values[i]['beta'], requires_grad=True))
        #'''

    def forward(self, x):
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = {
                'gamma': self.gammas[block],
                'beta': self.betas[block]
            }
            block_params.append(block_param_dict)
        return block_params


class FilmAdapter(nn.Module):
    def __init__(self, layer, adaptation_config, task_dim=None, init_films=None):
        super().__init__()
        self.num_maps = adaptation_config['num_maps_per_layer']
        self.num_blocks = adaptation_config['num_blocks_per_layer']
        self.task_dim = task_dim
        self.num_target_layers = len(self.num_maps)
        self.layer = layer
        self.num_generated_params = 0
        self.layers = self.get_layers(init_films)

    def get_layers(self, init_films):
        layers = nn.ModuleList()
        if init_films is None:
            for num_maps, num_blocks in zip(self.num_maps, self.num_blocks):
                layers.append(
                    self.layer(
                        num_maps=num_maps,
                        num_blocks=num_blocks,
                        task_dim=self.task_dim
                    )
                )
                self.num_generated_params += layers[-1].num_generated_params
        else:
            count = 0
            for num_maps, num_blocks in zip(self.num_maps, self.num_blocks):
                layers.append(
                    self.layer(
                        num_maps=num_maps,
                        num_blocks=num_blocks,
                        task_dim=self.task_dim,
                        init_values=init_films[count]
                    )
                )
                count += 1
                self.num_generated_params += layers[-1].num_generated_params
        return layers

    def forward(self, x):
        return [self.layers[layer](x) for layer in range(self.num_target_layers)]


    def regularization_term(self):
        l2_term = 0
        for layer in self.layers:
            l2_term += layer.regularization_term()
        return l2_term


class DenseBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(DenseBlock, self).__init__()
        self.linear1 = nn.Linear(in_size, in_size)
        self.layernorm = nn.LayerNorm(in_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_size, out_size)

    def forward(self, x):
        out = self.linear1(x)
        out = self.layernorm(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out


class FilmLayerGenerator(BaseFilmLayer):
    def __init__(self, num_maps, num_blocks, task_dim):
        BaseFilmLayer.__init__(self, num_maps, num_blocks)
        self.task_dim = task_dim

        self.gamma_generators, self.gamma_regularizers = nn.ModuleList(), nn.ParameterList()
        self.beta_generators, self.beta_regularizers = nn.ModuleList(), nn.ParameterList()

        for i in range(self.num_blocks):
            self.num_generated_params += 2 * num_maps[i]
            self.gamma_generators.append(self._make_layer(self.task_dim, num_maps[i]))
            self.gamma_regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(num_maps[i]), 0, 0.001),
                                                        requires_grad=True))

            self.beta_generators.append(self._make_layer(self.task_dim, num_maps[i]))
            self.beta_regularizers.append(nn.Parameter(nn.init.normal_(torch.empty(num_maps[i]), 0, 0.001),
                                                       requires_grad=True))

    @staticmethod
    def _make_layer(in_size, out_size):
        return DenseBlock(in_size, out_size)

    def forward(self, x):
        """
        Forward pass through adaptation network.
        :param x: (torch.tensor) Input representation to network (task level representation z).
        :return: (list::dictionaries) Dictionary for every block in layer. Dictionary contains all the parameters
                 necessary to adapt layer in base network. Base network is aware of dict structure and can pull params
                 out during forward pass.
        """
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = {
                'gamma': self.gamma_generators[block](x).squeeze() * self.gamma_regularizers[block] +
                         torch.ones_like(self.gamma_regularizers[block]),
                'beta': self.beta_generators[block](x).squeeze() * self.beta_regularizers[block],
            }
            block_params.append(block_param_dict)
        return block_params


class DenseResidualLayer(nn.Module):
    """
    PyTorch like layer for standard linear layer with identity residual connection.
    :param num_features: (int) Number of input / output units for the layer.
    """
    def __init__(self, num_features):
        super(DenseResidualLayer, self).__init__()
        self.linear = nn.Linear(num_features, num_features)

    def forward(self, x):
        """
        Forward-pass through the layer. Implements the following computation:

                f(x) = f_theta(x) + x
                f_theta(x) = W^T x + b

        :param x: (torch.tensor) Input representation to apply layer to ( dim(x) = (batch, num_features) ).
        :return: (torch.tensor) Return f(x) ( dim(f(x) = (batch, num_features) ).
        """
        identity = x
        out = self.linear(x)
        out += identity
        return out

class DenseResidualBlock(nn.Module):
    """
    Wrapping a number of residual layers for residual block. Will be used as building block in FiLM hyper-networks.
    :param in_size: (int) Number of features for input representation.
    :param out_size: (int) Number of features for output representation.
    """
    def __init__(self, in_size, out_size):
        super(DenseResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_size, out_size)
        self.linear2 = nn.Linear(out_size, out_size)
        self.linear3 = nn.Linear(out_size, out_size)
        self.elu = nn.ELU()

    def forward(self, x):
        """
        Forward pass through residual block. Implements following computation:

                h = f3( f2( f1(x) ) ) + x
                or
                h = f3( f2( f1(x) ) )

                where fi(x) = Elu( Wi^T x + bi )

        :param x: (torch.tensor) Input representation to apply layer to ( dim(x) = (batch, in_size) ).
        :return: (torch.tensor) Return f(x) ( dim(f(x) = (batch, out_size) ).
        """
        identity = x
        out = self.linear1(x)
        out = self.elu(out)
        out = self.linear2(out)
        out = self.elu(out)
        out = self.linear3(out)
        if x.shape[-1] == out.shape[-1]:
            out += identity
        return out

class FilmAdaptationNetwork(nn.Module):
    """
    FiLM adaptation network (outputs FiLM adaptation parameters for all layers in a base feature extractor).
    :param layer: (FilmLayerNetwork) Layer object to be used for adaptation.
    :param num_maps_per_layer: (list::int) Number of feature maps for each layer in the network.
    :param num_blocks_per_layer: (list::int) Number of residual blocks in each layer in the network
                                 (see ResNet file for details about ResNet architectures).
    :param z_g_dim: (int) Dimensionality of network input. For this network, z is shared across all layers.
    """
    def __init__(self, layer, num_maps_per_layer, num_blocks_per_layer, z_g_dim):
        super().__init__()
        self.z_g_dim = z_g_dim
        self.num_maps = num_maps_per_layer
        self.num_blocks = num_blocks_per_layer
        self.num_target_layers = len(self.num_maps)
        self.layer = layer
        self.layers = self.get_layers()

    def get_layers(self):
        """
        Loop over layers of base network and initialize adaptation network.
        :return: (nn.ModuleList) ModuleList containing the adaptation network for each layer in base network.
        """
        layers = nn.ModuleList()
        for num_maps, num_blocks in zip(self.num_maps, self.num_blocks):
            layers.append(
                self.layer(
                    num_maps=num_maps,
                    num_blocks=num_blocks,
                    z_g_dim=self.z_g_dim
                )
            )
        return layers

    def forward(self, x):
        """
        Forward pass through adaptation network to create list of adaptation parameters.
        :param x: (torch.tensor) (z -- task level representation for generating adaptation).
        :return: (list::adaptation_params) Returns a list of adaptation dictionaries, one for each layer in base net.
        """
        return [self.layers[layer](x) for layer in range(self.num_target_layers)]

    def regularization_term(self):
        """
        Simple function to aggregate the regularization terms from each of the layers in the adaptation network.
        :return: (torch.scalar) A order-0 torch tensor with the regularization term for the adaptation net params.
        """
        l2_term = 0
        for layer in self.layers:
            l2_term += layer.regularization_term()
        return l2_term

class FilmLayerNetwork(nn.Module):
    """
    Single adaptation network for generating the parameters of each layer in the base network. Will be wrapped around
    by FilmAdaptationNetwork.
    :param num_maps: (int) Number of output maps to be adapted in base network layer.
    :param num_blocks: (int) Number of blocks being adapted in the base network layer.
    :param z_g_dim: (int) Dimensionality of input to network (task level representation).
    """
    def __init__(self, num_maps, num_blocks, z_g_dim):
        super().__init__()
        self.z_g_dim = z_g_dim
        self.num_maps = num_maps
        self.num_blocks = num_blocks

        # Initialize a simple shared layer for all parameter adapters (gammas and betas)
        self.shared_layer = nn.Sequential(
            nn.Linear(self.z_g_dim, self.num_maps),
            nn.ReLU()
        )

        # Initialize the processors (adaptation networks) and regularization lists for each of the output params
        self.gamma1_processors, self.gamma1_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()
        self.gamma2_processors, self.gamma2_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()
        self.beta1_processors, self.beta1_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()
        self.beta2_processors, self.beta2_regularizers = torch.nn.ModuleList(), torch.nn.ParameterList()

        # Generate the required layers / regularization parameters, and collect them in ModuleLists and ParameterLists
        for _ in range(self.num_blocks):
            regularizer = torch.nn.init.normal_(torch.empty(num_maps), 0, 0.001)

            self.gamma1_processors.append(self._make_layer(num_maps))
            self.gamma1_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))

            self.beta1_processors.append(self._make_layer(num_maps))
            self.beta1_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))

            self.gamma2_processors.append(self._make_layer(num_maps))
            self.gamma2_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))

            self.beta2_processors.append(self._make_layer(num_maps))
            self.beta2_regularizers.append(torch.nn.Parameter(regularizer, requires_grad=True))

    @staticmethod
    def _make_layer(size):
        """
        Simple layer generation method for adaptation network of one of the parameter sets (all have same structure).
        :param size: (int) Number of parameters in layer.
        :return: (nn.Sequential) Three layer dense residual network to generate adaptation parameters.
        """
        return nn.Sequential(
            DenseResidualLayer(size),
            nn.ReLU(),
            DenseResidualLayer(size),
            nn.ReLU(),
            DenseResidualLayer(size)
        )

    def forward(self, x):
        """
        Forward pass through adaptation network.
        :param x: (torch.tensor) Input representation to network (task level representation z).
        :return: (list::dictionaries) Dictionary for every block in layer. Dictionary contains all the parameters
                 necessary to adapt layer in base network. Base network is aware of dict structure and can pull params
                 out during forward pass.
        """
        x = self.shared_layer(x)
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = {
                'gamma1': self.gamma1_processors[block](x).squeeze() * self.gamma1_regularizers[block] +
                          torch.ones_like(self.gamma1_regularizers[block]),
                'beta1': self.beta1_processors[block](x).squeeze() * self.beta1_regularizers[block],
                'gamma2': self.gamma2_processors[block](x).squeeze() * self.gamma2_regularizers[block] +
                          torch.ones_like(self.gamma2_regularizers[block]),
                'beta2': self.beta2_processors[block](x).squeeze() * self.beta2_regularizers[block]
            }
            block_params.append(block_param_dict)
        return block_params

    def regularization_term(self):
        """
        Compute the regularization term for the parameters. Recall, FiLM applies gamma * x + beta. As such, params
        gamma and beta are regularized to unity, i.e. ||gamma - 1||_2 and ||beta||_2.
        :return: (torch.tensor) Scalar for l2 norm for all parameters according to regularization scheme.
        """
        l2_term = 0
        for gamma_regularizer, beta_regularizer in zip(self.gamma1_regularizers, self.beta1_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        for gamma_regularizer, beta_regularizer in zip(self.gamma2_regularizers, self.beta2_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        return l2_term


class BaseFilmLayerResNet(nn.Module):
    """
    Base class for a FiLM layer in a ResNet feature extractor. Will be wrapped around a FilmAdapter instance.
    """
    def __init__(self, num_maps, num_blocks):
        """
        Creates a BaseFilmLayer instance.
        :param num_maps: (int) Dimensionality of input to each block in the FiLM layer.
        :param num_blocks: (int) Number of blocks in the FiLM layer.
        :return: Nothing.
        """
        super(BaseFilmLayerResNet, self).__init__()
        
        self.num_maps = num_maps
        self.num_blocks = num_blocks
        self.num_layers_per_block = 2
        self.num_generated_params = 0

class FilmLayerResNet(BaseFilmLayerResNet):
    """
    Class for a learnable FiLM layer in an EfficientNet feature extractor. Here, the FiLM layer is a set of nn.ModuleList() made up of nn.ParameterList()s made up of nn.Parameter()s, which are updated via standard gradient steps.
    """
    def __init__(self, num_maps, num_blocks, task_dim=None):
        """
        Creates a FilmLayer instance.
        :param num_maps: (int) Dimensionality of input to each block in the FiLM layer.
        :param num_blocks: (int) Number of blocks in the FiLM layer.
        :param task_dim: (None) Not used.
        :return: Nothing.
        """
        BaseFilmLayerResNet.__init__(self, num_maps, num_blocks)
        self._init_layer()

    def _init_layer(self):
        """
        Function that creates and initialises the FiLM layer. The FiLM layer has a nn.ModuleList() for its gammas and betas (and their corresponding regularisers). Each element in a nn.ModuleList() is a nn.ParamaterList() and correspondings to one block in the FiLM layer. Each element in a nn.ParameterList() is a nn.Parameter() of size self.num_maps and corresponds to a layer in the block.
        :return: Nothing.
        """
        # Initialize the gamma/beta lists for each of the output params
        self.gamma1, self.beta1 = torch.nn.ParameterList(), torch.nn.ParameterList()
        self.gamma2, self.beta2 = torch.nn.ParameterList(), torch.nn.ParameterList()

        # Generate the required layers / regularization parameters, and collect them in ModuleLists and ParameterLists
        for _ in range(self.num_blocks):
            self.gamma1.append(nn.Parameter(torch.ones(self.num_maps), requires_grad=True))
            self.beta1.append(nn.Parameter(torch.zeros(self.num_maps), requires_grad=True))
            self.gamma2.append(nn.Parameter(torch.ones(self.num_maps), requires_grad=True))
            self.beta2.append(nn.Parameter(torch.zeros(self.num_maps), requires_grad=True))
        

    def forward(self, x=None):
        """
        Function that returns the FiLM layer's parameters. Note, input x is ignored.
        :param x: (None) Not used.
        :return: (list::dict::list::nn.Parameter) Parameters of the FiLM layer.
        """
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = {
                'gamma1': self.gamma1[block],
                'beta1': self.beta1[block],
                'gamma2': self.gamma2[block],
                'beta2': self.beta2[block]
            }
            block_params.append(block_param_dict)
        return block_params   
