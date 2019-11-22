from collections import namedtuple

Genotype = namedtuple('Genotype', 'recurrent concat')

PRIMITIVES = [
    'none',
    'tanh',
    'relu',
    'sigmoid',
    'identity'
]

# number of intermediate nodes

ENAS = Genotype(
    recurrent = [
        ('tanh', 0),
        ('tanh', 1),
        ('relu', 2),
        ('tanh', 3),
        ('tanh', 3),
        ('relu', 3),
        ('relu', 4),
        ('relu', 7),
        ('relu', 8),
        ('relu', 8),
        ('relu', 8),
        ('tanh', 2),
    ],
    concat = [5, 6, 9, 10, 11, 12]
)

ENAS_wpl = Genotype(
        recurrent = [
            ('relu', 0),
            ('sigmoid', 1),
            ('relu', 1),
            ('relu', 1),
            ('sigmoid', 2),
            ('sigmoid', 4),
            ('relu', 4),
            ('sigmoid', 4),
            ('relu', 4),
            ('tanh', 4),
            ('tanh', 5),
            ('relu', 11)
        ],
        concat = [3, 6, 7, 8, 9, 10, 12]
    )

DARTS_V1 = Genotype(
    recurrent=[
        ('relu', 0),
        ('relu', 1),
        ('tanh', 2),
        ('relu', 3),
        ('relu', 4),
        ('identity', 1),
        ('relu', 5),
        ('relu', 1)
    ],
    concat=range(1, 9))

DARTS_V2 = Genotype(recurrent=[('sigmoid', 0), ('relu', 1), ('relu', 1), ('identity', 1), ('tanh', 2), ('sigmoid', 5), ('tanh', 3), ('relu', 5)], concat=range(1, 9))

DARTS = DARTS_V2

PSO = Genotype(recurrent=[('sigmoid', 0), ('identity', 0), ('tanh', 0), ('tanh', 2), ('sigmoid', 1), ('sigmoid', 3), ('identity', 0), ('identity', 1)], concat=range(1, 9))

pso_no_lin_32_4_no_handle = Genotype(recurrent=[('sigmoid', 0), ('sigmoid', 0), ('relu', 0), ('relu', 1), ('relu', 0), ('relu', 3), ('tanh', 0), ('relu', 2)], concat=range(1, 9))

pso_NORM_32_18 = Genotype(recurrent=[('sigmoid', 0), ('identity', 0), ('sigmoid', 0), ('identity', 0), ('tanh', 3), ('tanh', 2), ('sigmoid', 2), ('relu', 2)], concat=range(1, 9))

pso_tanh_32_18 = Genotype(recurrent=[('sigmoid', 0), ('relu', 1), ('tanh', 2), ('sigmoid', 3), ('tanh', 2), ('tanh', 0), ('identity', 1), ('identity', 1)], concat=range(1, 9))

pso_tanh_256_4_abs_max = Genotype(recurrent=[('tanh', 0), ('tanh', 0), ('identity', 0), ('tanh', 3), ('tanh', 1), ('sigmoid', 1), ('relu', 1), ('identity', 3)], concat=range(1, 9))

pso_tanh_32_18_abs_max = Genotype(recurrent=[('sigmoid', 0), ('identity', 0), ('identity', 2), ('sigmoid', 3), ('relu', 2), ('identity', 2), ('relu', 0), ('sigmoid', 0)], concat=range(1, 9))

pso_NEW_INIT_TANH_BS32_NB18_P46 = Genotype(recurrent=[('identity', 0), ('identity', 0), ('tanh', 2), ('sigmoid', 0), ('relu', 1), ('identity', 4), ('identity', 1), ('sigmoid', 7)], concat=range(1, 9))

pso_CLUSTERS_TANH_EPOCH_41 = Genotype(recurrent=[('identity', 0), ('sigmoid', 1), ('sigmoid', 0), ('sigmoid', 0), ('sigmoid', 0), ('relu', 0), ('tanh', 0), ('tanh', 2)], concat=range(1, 9))

pso_clusters_bs_256 = Genotype(recurrent=[('tanh', 0), ('tanh', 1), ('relu', 0), ('identity', 0), ('tanh', 0), ('relu', 0), ('tanh', 0), ('identity', 0)], concat=range(1, 9))

pso_slot_tanh_bs_32_no_init = Genotype(recurrent=[('identity', 0), ('identity', 0), ('relu', 0), ('relu', 1), ('tanh', 2), ('identity', 3), ('tanh', 6), ('relu', 0)], concat=range(1, 9))

pso_slot_tanh_bs_32_no_init_pop_207 = Genotype(recurrent=[('identity', 0), ('relu', 0), ('tanh', 0), ('identity', 0), ('tanh', 0), ('tanh', 3), ('tanh', 3), ('identity', 1)], concat=range(1, 9))

pso_no_lin_no_relu_bs_no_slot_46 = Genotype(recurrent=[('sigmoid', 0), ('tanh', 1), ('tanh', 0), ('tanh', 0), ('tanh', 0), ('sigmoid', 2), ('sigmoid', 0), ('tanh', 0)], concat=range(1, 9))

pso_slot_pop50_nodes4_tanh = Genotype(recurrent=[('tanh', 0), ('identity', 0), ('identity', 0), ('identity', 0)], concat=range(1, 5))

pso_slot_pop50_node8_tanh_seed_1267 = Genotype(recurrent=[('tanh', 0), ('tanh', 0), ('relu', 0), ('tanh', 0), ('identity', 2), ('tanh', 0), ('tanh', 0), ('identity', 4)], concat=range(1, 9))

wpl_pso_slot46_node8_tanh_seed_3 = Genotype(recurrent=[('tanh', 0), ('tanh', 0), ('tanh', 0), ('tanh', 1), ('identity', 3), ('identity', 1), ('tanh', 1), ('tanh', 1)], concat=range(1, 9))

pso_slot46_node8_tanh_seed_3_start_stop = Genotype(recurrent=[('sigmoid', 0), ('tanh', 1), ('tanh', 0), ('relu', 3), ('relu', 0), ('identity', 1), ('relu', 5), ('identity', 3)], concat=range(1, 9))

pso_slot46_node8_tanh_seed_3_updates_20 = Genotype(recurrent=[('identity', 0), ('tanh', 0), ('sigmoid', 2), ('relu', 1), ('tanh', 2), ('sigmoid', 4), ('tanh', 2), ('sigmoid', 2)], concat=range(1, 9))

pso_slot46_node8_tanh_seed_3_updates_5_coeff_025_06_1 = Genotype(recurrent=[('identity', 0), ('relu', 1), ('sigmoid', 2), ('sigmoid', 0), ('identity', 4), ('tanh', 2), ('tanh', 1), ('tanh', 6)], concat=range(1, 9))

pso_pretrained_node8_tanh_seed_3_updates_5 = Genotype(recurrent=[('tanh', 0), ('tanh', 1), ('relu', 0), ('tanh', 1), ('identity', 2), ('identity', 1), ('sigmoid', 2), ('tanh', 1)], concat=range(1, 9))

##### TANH BEST DIFFERENT SEED 10E-3
pso_start0_steps0_tanh_seed42 = Genotype(recurrent=[('tanh', 0), ('identity', 0), ('tanh', 2), ('sigmoid', 3), ('relu', 2), ('tanh', 0), ('identity', 1), ('tanh', 1)], concat=range(1, 9))
pso_start0_steps0_tanh_seed19950505 = Genotype(recurrent=[('tanh', 0), ('identity', 1), ('relu', 0), ('sigmoid', 1), ('identity', 0), ('tanh', 1), ('identity', 0), ('sigmoid', 0)], concat=range(1, 9))
pso_start0_steps0_tanh_seed20060709 = Genotype(recurrent=[('tanh', 0), ('tanh', 1), ('identity', 1), ('relu', 0), ('relu', 3), ('relu', 1), ('identity', 1), ('tanh', 1)], concat=range(1, 9))

#### TANH SLOT START 2
pso_start2_step0_slot_tanh_seed42 = Genotype(recurrent=[('relu', 0), ('identity', 1), ('sigmoid', 0), ('sigmoid', 2), ('identity', 0), ('relu', 4), ('tanh', 1), ('tanh', 5)], concat=range(1, 9))

####################################
####################################
####################################
########## FINAL SEARCHES ##########

#### DARTS SECOND ORDER SEEDS ####
darts_second_order_seed3 = Genotype(recurrent=[('sigmoid', 0), ('relu', 1), ('relu', 1), ('identity', 1), ('tanh', 2), ('sigmoid', 5), ('tanh', 3), ('relu', 5)], concat=range(1, 9))
darts_second_order_seed42 = Genotype(recurrent=[('sigmoid', 0), ('relu', 0), ('tanh', 1), ('sigmoid', 2), ('identity', 0), ('identity', 1), ('identity', 1), ('relu', 1)], concat=range(1, 9))
darts_second_order_seed1995 = Genotype(recurrent=[('tanh', 0), ('relu', 0), ('tanh', 1), ('tanh', 1), ('tanh', 2), ('relu', 5), ('tanh', 2), ('tanh', 1)], concat=range(1, 9))
darts_second_order_seed2006 = Genotype(recurrent=[('sigmoid', 0), ('sigmoid', 0), ('sigmoid', 2), ('relu', 0), ('sigmoid', 4), ('sigmoid', 4), ('relu', 6), ('relu', 1)], concat=range(1, 9))

#### TANH START 2 SLOTS GLOROT NEW MATRICES PSO STEPS 1 W=0.5 C1=1 C2=2 ####
pso_start2_step0_slot_tanh_sd3_w05_cl1_cg2 = Genotype(recurrent=[('tanh', 0), ('relu', 1), ('sigmoid', 1), ('sigmoid', 1), ('relu', 4), ('tanh', 0), ('identity', 0), ('tanh', 6)], concat=range(1, 9))
pso_start2_step0_slot_tanh_sd42_w05_cl1_cg2 = Genotype(recurrent=[('sigmoid', 0), ('sigmoid', 0), ('relu', 2), ('relu', 1), ('tanh', 3), ('identity', 4), ('tanh', 5), ('sigmoid', 2)], concat=range(1, 9))
pso_start2_step0_slot_tanh_sd1995_w05_cl1_cg2 = Genotype(recurrent=[('tanh', 0), ('sigmoid', 0), ('sigmoid', 0), ('identity', 1), ('identity', 3), ('sigmoid', 1), ('identity', 0), ('sigmoid', 0)], concat=range(1, 9))
pso_start2_step0_slot_tanh_sd2006_w05_cl1_cg2 = Genotype(recurrent=[('tanh', 0), ('identity', 1), ('identity', 0), ('tanh', 0), ('relu', 3), ('tanh', 2), ('tanh', 4), ('sigmoid', 1)], concat=range(1, 9))

#### TANH START 2 SLOTS GLOROT NEW MATRICES PSO STEPS 1 W=0.65 C1=1 C2=2 ####
pso_start2_step0_slot_tanh_sd3_w065_cl1_cg2 = Genotype(recurrent=[('relu', 0), ('relu', 0), ('tanh', 2), ('relu', 0), ('relu', 0), ('sigmoid', 0), ('tanh', 2), ('identity', 1)], concat=range(1, 9))
pso_start2_step0_slot_tanh_sd42_w065_cl1_cg2 = Genotype(recurrent=[('relu', 0), ('identity', 0), ('identity', 0), ('sigmoid', 0), ('sigmoid', 1), ('tanh', 4), ('identity', 1), ('identity', 2)], concat=range(1, 9))
pso_start2_step0_slot_tanh_sd1995_w065_cl1_cg2 = Genotype(recurrent=[('tanh', 0), ('identity', 0), ('sigmoid', 0), ('sigmoid', 2), ('identity', 1), ('identity', 3), ('tanh', 4), ('identity', 0)], concat=range(1, 9))
pso_start2_step0_slot_tanh_sd2006_w065_cl1_cg2 = Genotype(recurrent=[('tanh', 0), ('tanh', 1), ('tanh', 2), ('sigmoid', 3), ('sigmoid', 2), ('identity', 1), ('sigmoid', 0), ('sigmoid', 1)], concat=range(1, 9))

#### tanh new space start 5 ####
pso_start5_step0_slot_tanh_sd3_w05_cl1_cg2_selu = Genotype(recurrent=[('sigmoid', 0), ('sigmoid', 0), ('selu', 2), ('sigmoid', 0), ('tanh', 1), ('selu', 5), ('tanh', 2), ('tanh', 7)], concat=range(1, 9))
pso_start5_step0_slot_tanh_sd42_w05_cl1_cg2_selu = Genotype(recurrent=[('tanh', 0), ('tanh', 0), ('tanh', 2), ('tanh', 1), ('tanh', 1), ('sigmoid', 2), ('selu', 6), ('tanh', 1)], concat=range(1, 9))
pso_start5_step0_slot_tanh_sd1995_w05_cl1_cg2_selu = Genotype(recurrent=[('tanh', 0), ('tanh', 0), ('sigmoid', 2), ('tanh', 3), ('sigmoid', 2), ('selu', 1), ('tanh', 3), ('selu', 4)], concat=range(1, 9))
pso_start5_step0_slot_tanh_sd2006_w05_cl1_cg2_selu = Genotype(recurrent=[('sigmoid', 0), ('tanh', 0), ('selu', 0), ('selu', 1), ('selu', 3), ('sigmoid', 1), ('selu', 0), ('sigmoid', 1)], concat=range(1, 9))

#### tanh new space start 4 updates 5 ####
pso_start5_step5_slot_tanh_sd3_w05_cl1_cg2_selu = Genotype(recurrent=[('tanh', 0), ('sigmoid', 1), ('sigmoid', 0), ('selu', 1), ('selu', 2), ('sigmoid', 0), ('tanh', 3), ('tanh', 2)], concat=range(1, 9))
pso_start5_step5_slot_tanh_sd42_w05_cl1_cg2_selu = Genotype(recurrent=[('tanh', 0), ('tanh', 0), ('tanh', 0), ('tanh', 0), ('tanh', 0), ('selu', 5), ('sigmoid', 5), ('tanh', 0)], concat=range(1, 9))
pso_start5_step5_slot_tanh_sd1995_w05_cl1_cg2_selu = Genotype(recurrent=[('tanh', 0), ('tanh', 0), ('tanh', 1), ('sigmoid', 2), ('tanh', 0), ('tanh', 1), ('selu', 0), ('sigmoid', 0)], concat=range(1, 9))
pso_start5_step5_slot_tanh_sd2006_w05_cl1_cg2_selu = Genotype(recurrent=[('tanh', 0), ('tanh', 0), ('tanh', 0), ('selu', 1), ('tanh', 0), ('sigmoid', 1), ('tanh', 3), ('tanh', 3)], concat=range(1, 9))

#### random cells old space ####
random_seed3 = Genotype(recurrent=[('identity', 0), ('relu', 0), ('sigmoid', 1), ('identity', 0), ('sigmoid', 0), ('relu', 3), ('sigmoid', 6), ('tanh', 3)], concat=range(1, 9))
random_seed1995 = Genotype(recurrent=[('identity', 0), ('relu', 1), ('sigmoid', 0), ('sigmoid', 2), ('sigmoid', 0), ('relu', 1), ('tanh', 0), ('sigmoid', 4)], concat=range(1, 9))
random_seed2006 = Genotype(recurrent=[('tanh', 0), ('tanh', 1), ('identity', 0), ('tanh', 1), ('sigmoid', 0), ('identity', 5), ('sigmoid', 1), ('tanh', 5)], concat=range(1, 9))

#### eval configs on search ####
eval_pso_seed_1267 = Genotype(recurrent=[('sigmoid', 0), ('identity', 0), ('relu', 0), ('sigmoid', 2), ('tanh', 2), ('identity', 0), ('relu', 4), ('identity', 0)], concat=range(1, 9))

#### EMPIRIC ####

EMP = Genotype(
    recurrent=[
        ('sigmoid', 0),
        ('sigmoid', 1),
        ('tanh', 2),
        ('sigmoid', 3),
        ('sigmoid', 4),
        ('identity', 1),
        ('sigmoid', 5),
        ('sigmoid', 1)
    ],
    concat=range(1, 9))


def get_forced_genotype_by_name(name):
    return globals()[name]
