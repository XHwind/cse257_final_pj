from sgm.dependencies import *
import tensorflow as tf
# tf.reset_default_graph()
tf.compat.v1.reset_default_graph()
import pdb
from sgm.envs import env_load_fn
from sgm.agent import UvfAgent
from sgm.envs import *
from scipy.spatial import distance
from sgm.evaluation import cleanup_and_eval 
from sgm.planner import SGMSearchPolicy
import pdb
def load_env(max_episode_steps, env_name, resize_factor, thin):


    tf_env = env_load_fn(env_name, max_episode_steps,
                         resize_factor=resize_factor,
                         terminate_on_timeout=False,
                         thin=thin)
    eval_tf_env = env_load_fn(env_name, max_episode_steps,
                              resize_factor=resize_factor,
                              terminate_on_timeout=True,
                              thin=thin)
    return tf_env, eval_tf_env


def main():
    max_episode_steps = 20
    env_name = 'FourRooms'  # Choose one of the environments shown above. 
    resize_factor = 5  # Inflate the environment to increase the difficulty.
    thin = True # If True, resize by expanding open space, not walls, to make walls thin
    tf_env, eval_tf_env = load_env(max_episode_steps, env_name, resize_factor, thin)
    desc_name = "thinned_" + env_name.lower() if thin else env_name.lower() 
    base_dir = os.path.join(os.getcwd(), os.pardir, "agents")
    model_dir = os.path.join(base_dir, desc_name)

    agent = UvfAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        max_episode_steps=max_episode_steps,
        use_distributional_rl=True,
        ensemble_size=3)

    filename = "FourRooms_coordinate_20steps-Dec-05-2019-12-58-28-PM/ckpt/"
    checkpoint_file = os.path.join(model_dir,    filename)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=agent)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_file, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    #@title Fill the replay buffer with random data  {vertical-output: true, run: "auto" }
    replay_buffer_size = 100 #@param {min:100, max: 1000, step: 100, type:"slider"}

    eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
        prob_constraint=0.0,
        min_dist=0,
        max_dist=np.inf)
    rb_vec = []
    for it in range(replay_buffer_size):
        ts = eval_tf_env.reset()
        rb_vec.append(ts.observation['observation'].numpy()[0])
    rb_vec = np.array(rb_vec)
    # plt.figure(figsize=(6, 6))
    # plt.scatter(*rb_vec.T)
    # plot_walls(eval_tf_env.pyenv.envs[0].env.walls)
    # plt.show()
    pdist = agent._get_pairwise_dist(rb_vec, aggregate=None).numpy()
    print(pdist)
    steps_total = []
    success_rate_total = []
    for search_policy_type in ["SGM", "SoRB"]:
        agent.initialize_search(rb_vec, max_search_steps=10) # needed to set agent attributes that SGMSearchPolicy constructor queries
        search_policy = SGMSearchPolicy(agent, pdist, rb_vec, rb_vec, cache_pdist = True)

        logdir = os.path.join(os.getcwd(), os.pardir, "logs")
        eval_difficulty = 0.95
        k_nearest = 5
        eval_trials = 10
        total_cleanup_steps = 100000
        eval_period = 5000
        logfolder, success_rate_list = cleanup_and_eval(search_policy,
                                     search_policy_type,
                                     desc_name,
                                     eval_tf_env,
                                     logdir,
                                     eval_difficulty=eval_difficulty,
                                     k_nearest=k_nearest,
                                     eval_trials=eval_trials,
                                     total_cleanup_steps=total_cleanup_steps,
                                     eval_period=eval_period,
                                     verbose=True)
        steps = [x[0] for x in success_rate_list]
        success_rate_list = [x[1] for x in success_rate_list]
        steps_total.append(steps)
        success_rate_total.append(success_rate_list)
    plt.plot(steps_total[0], success_rate_total[0])
    plt.plot(steps_total[1], success_rate_total[1])
    plt.show()


if __name__=="__main__":
    main()  