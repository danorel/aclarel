import argparse

import environments.mountain_car.environment as mountain_car 
import environments.mountain_car.experiments as experiments

def main():
    parser = argparse.ArgumentParser(
        prog='Curriculum Learning method in MountainCar environment',
        description='Runs an agent via Curriculum Learning',
        epilog='Enjoy and experiment!'
    )
    parser.add_argument('-a', '--agent_name', choices=['dqn', 'q-learning'], default='dqn', help='Choose the agent type: dqn or q-learning')
    parser.add_argument('-c', '--curriculum_name', choices=['baseline', 'one-pass', 'root-p', 'hard', 'linear', 'logarithmic', 'logistic', 'mixture', 'polynomial', 'anti-curriculum', 'transfer-learning', 'teacher-learning'], default='baseline', help='Select the curriculum method')
    parser.add_argument('-p', '--use_pretrained', action='store_true', help='Flag whether use a pre-trained agent or not')
    parser.add_argument('-r', '--use_render', action='store_true', help='Flag whether render environment or not')

    args = parser.parse_args()

    print(f"Configuring RL agent '{args.agent_name}' and CL method '{args.curriculum_name}'")
    agent = experiments.get_agent(args.agent_name, args.curriculum_name, args.use_pretrained)
    curriculum = experiments.get_curriculum(agent)

    print(f"Running RL agent '{args.agent_name}' via CL method '{args.curriculum_name}'")
    mountain_car.train_evaluate(
        agent=agent,
        curriculum=curriculum, 
        use_render=args.use_render
    )

if __name__ == "__main__":
    main()