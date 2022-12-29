<<<<<<< HEAD
<<<<<<< HEAD
import json
from sklearn.pipeline import Pipeline, FeatureUnion

# Relative imports
from pyrcn.base.blocks import InputToNode, NodeToNode
from pyrcn.echo_state_network import ESNRegressor
=======
from pyrcn.base.blocks import InputToNode, NodeToNode
from pyrcn.echo_state_network import ESNRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
>>>>>>> 1cf6832f6b18363625f35930ef44e76dc778b510
=======
from pyrcn.base.blocks import InputToNode, NodeToNode
from pyrcn.echo_state_network import ESNRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
>>>>>>> 1cf6832f6b18363625f35930ef44e76dc778b510

def Vanilla_input2node(I2N_config):
    return InputToNode(
        hidden_layer_size=I2N_config["layer_size"], k_in=I2N_config["sparsity"], input_activation=I2N_config["activation"],
        input_scaling=I2N_config["scaling"], input_shift=I2N_config["shift"], bias_scaling=I2N_config["bias_scaling"],
        bias_shift=I2N_config["bias_shift"], random_state=I2N_config["random_seed"]
    )
    
def Vanilla_node2node(N2N_config):
    return NodeToNode(
        hidden_layer_size=N2N_config["layer_size"], sparsity=N2N_config["sparsity"], reservoir_activation=N2N_config["activation"],
        spectral_radius=N2N_config["spectral_radius"], leakage=N2N_config["leakage"], bidirectional=N2N_config["bidirectional"],
        random_state=N2N_config["random_seed"]
    )

<<<<<<< HEAD
<<<<<<< HEAD
def Sequential_block(I2N_config, N2N_config, blocks=2):
=======
def Sequential_block(I2N_config, N2N_config, blocks=1):
>>>>>>> 1cf6832f6b18363625f35930ef44e76dc778b510
=======
def Sequential_block(I2N_config, N2N_config, blocks=1):
>>>>>>> 1cf6832f6b18363625f35930ef44e76dc778b510
    input2node = Vanilla_input2node(I2N_config)
    pipeline = [('i2n', input2node)]
    for i in range(blocks):
        node2node = Vanilla_node2node(N2N_config)
        pipeline.append(('n2n' + str(i+1), node2node))
    return Pipeline(pipeline)

def Parallel_block(I2N_config, N2N_config, blocks=2):
    pipeline = []
    for i in range(blocks):
        block = Sequential_block(I2N_config, N2N_config)
        pipeline.append(('block_'+str(i+1), block))
    return FeatureUnion(pipeline)
    

def reservoir_network(I2N, N2N):
<<<<<<< HEAD
<<<<<<< HEAD
    """
    TODO: Add description
    https://pyrcn.readthedocs.io/en/main/api/pyrcn.base.html

    Arguments
    ---------
    I2N: (pyRCN InputToNode object) 
    N2N: (pyRCN NodeToNode object)

    Outputs
    ---------
    ESNRegressor: (pyRCN ESNRegressor object) 
    """

    return ESNRegressor(input_to_node=I2N, node_to_node=N2N)

def return_reservoir_blocks(json_file, exec_args):
    """
    TODO: Add description
    https://pyrcn.readthedocs.io/en/main/api/pyrcn.base.html

    Arguments
    ---------
    json_config: (string) json file name with the parameters of the reservoir blocks 
    opts: (argsparse object) Object containing the command line arguments

    Outputs
    ---------
    I2N: (pyRCN InputToNode object) 
    N2N: (pyRCN NodeToNode object)
    """

    json_config = open(json_file, 'r')
    params = json.load(json_config)
    json_config.close()
    if exec_args.blocks == "vanilla": 
        I2N = Vanilla_input2node(params["Input_2_Node"])
        N2N = Vanilla_node2node(params["Node_2_Node"])
    elif exec_args.blocks == "sequential":
        assert exec_args.num_blocks is not int, "Please provide the number of Sequential blocks with the -nb or --num_blocks argument"
        I2N = Sequential_block(params["Input_2_Node"], params["Node_2_Node"], blocks=exec_args.num_blocks)
        N2N = Vanilla_node2node(params["Node_2_Node"])
    else:
        assert exec_args.num_blocks is not int, "Please provide the number of Parallel blocks with the -nb or --num_blocks argument"
        I2N = Parallel_block(params["Input_2_Node"], params["Node_2_Node"], blocks=exec_args.num_blocks)
        params["Node_2_Node"]["layer_size"] = exec_args.num_blocks * params["Node_2_Node"]["layer_size"]
        N2N = Vanilla_node2node(params["Node_2_Node"])

    return I2N, N2N
    
if __name__ == '__main__':
    pass
=======
    return ESNRegressor(input_to_node=I2N, node_to_node=N2N)
>>>>>>> 1cf6832f6b18363625f35930ef44e76dc778b510
=======
    return ESNRegressor(input_to_node=I2N, node_to_node=N2N)
>>>>>>> 1cf6832f6b18363625f35930ef44e76dc778b510
