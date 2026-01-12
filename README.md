# SAC for Medical Robotics

Reinforcement learning for robotic surgery. Trains a robot to handle tissues with different stiffness levels.

## What This Does

Trains a Soft Actor-Critic (SAC) agent to reach a target position (20cm) across different tissue stiffness values (50-600 N/m). Compares 5 different training strategies to see which generalizes best to new, unseen stiffness values.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete training and evaluation pipeline:

```bash
python train.py
```

This will:
- Train 5 different protocols
- Test on unseen stiffness values
- Generate comparison plots
- Save all results

Takes about 10 hours to complete.

## Training Protocols

**Linear Small**: Small steps (40 N/m increments)
- Trains on 15 stiffness levels
- Best performance on unseen values

**Linear Large**: Large steps (110 N/m increments)
- Trains on 6 stiffness levels
- Fastest training but worse generalization

**Logarithmic**: More practice on easy cases
- Trains on 15 levels with logarithmic spacing

**Random**: Random stiffness values
- Baseline comparison

**Patient Categories**: Grouped by difficulty
- Easy, normal, and difficult patients

## Results

After running, you get:

**Models**: `models_challenging/*.zip` - Trained agents
**Plots**: `plots_challenging/*.png` - 4 comparison figures
**Data**: `results_challenging/*.json` - All metrics

## Main Finding

Small training steps generalize better than large steps, even though they take longer to train. The agent learns to interpolate between training points rather than just memorizing specific values.

## Environment Details

The agent controls a 1D spring-mass system:
- Applies force to move a mass
- Spring pulls back based on stiffness
- Goal: reach 20cm with 1cm precision

State: position, velocity, target distance, stiffness, time
Action: continuous force (-200 to 200 N)
Physics: Hooke's law (F = -kx)

## Citation

```bibtex
@article{vandebeek2025sac,
  title={Reinforcement Learning in Robotic Surgery: Using Soft Actor-Critic Algorithms for Adaptive Tissue Handling},
  author={van den Boogaart Castro, Lucia},
  year={2025}
}
```

## Contact

Lucia van de Beek
GitHub: [@luciavdbc](https://github.com/luciavdbc)

## License

MIT License - see LICENSE file
