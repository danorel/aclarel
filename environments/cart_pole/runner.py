import argparse

import environments.cart_pole.environment as cart_pole
import environments.cart_pole.experiments as experiments

def main():
    parser = argparse.ArgumentParser(
        prog='Curriculum Learning method',
        description='Runs an agent via Curriculum Learning',
        epilog='Enjoy and experiment!'
    )
    parser.add_argument('-a', '--agent_name', choices=['dqn', 'q-learning'], default='dqn', help='Choose the agent type: dqn or q-learning')
    parser.add_argument('-c', '--curriculum_name', choices=['baseline', 'one-pass', 'root-p', 'hard', 'linear', 'logarithmic', 'logistic', 'mixture', 'polynomial', 'anti-curriculum'], default='baseline', help='Select the curriculum method')
    parser.add_argument('-p', '--use_pretrained', action='store_true', help='Flag whether use a pre-trained agent or not')
    parser.add_argument('-r', '--use_render', action='store_true', help='Flag whether render environment or not')

    args = parser.parse_args()

    print(f"Running RL agent '{args.agent_name}' via CL method '{args.curriculum_name}'")
    cart_pole.train_evaluate(
        agent=experiments.get_agent(args.agent_name, args.curriculum_name, args.use_pretrained), 
        curriculum=experiments.get_curriculum(args.curriculum_name), 
        use_render=args.use_render
    )

if __name__ == "__main__":
    main()