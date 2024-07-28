import os
from itertools import combinations
from io import StringIO

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from github import Github
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory

load_dotenv()

app = Flask(__name__)

# GitHub configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
REPO_NAME = 'angelwshotgun/fbcs'
FILE_PATH = 'match_data.csv'

g = Github(GITHUB_TOKEN)
repo = g.get_repo(REPO_NAME)


def read_csv_from_github() -> pd.DataFrame:
    file_content = repo.get_contents(FILE_PATH)
    file_data = file_content.decoded_content.decode('utf-8')
    return pd.read_csv(StringIO(file_data))


def save_csv_to_github(data: pd.DataFrame) -> None:
    csv_content = data.to_csv(index=False)
    file = repo.get_contents(FILE_PATH)
    repo.update_file(FILE_PATH, "Update CSV file", csv_content, file.sha)


def initialize_data():
    global X, y, coefficients_df, coefficients_list, df

    np.random.seed(42)
    df = read_csv_from_github()
    X = df.drop(columns=['Result'])
    y = df['Result']

    coefficients_list = []

    for player in X.columns:
        player_data = df[df[player] != 0]

        if not player_data.empty:
            X_player = player_data.drop(columns=['Result'])
            y_player = player_data['Result']

            model = LogisticRegression(random_state=42)
            model.fit(X_player, y_player)

            coefficient = model.coef_[0][X_player.columns.get_loc(player)]
            coefficients_list.append((player, coefficient))

    coefficients_df = pd.DataFrame(coefficients_list, columns=['Player', 'Coefficient'])


initialize_data()  # Initialize the data when the application starts


def retrain_model() -> None:
    initialize_data()


def calculate_pair_coefficients(players: list) -> dict:
    np.random.seed(42)
    pair_coefficients = {}
    for p1, p2 in combinations(sorted(players), 2):
        matches = df[(df[p1] != 0) & (df[p2] != 0)]
        if not matches.empty and matches.shape[0] > 1:
            X = (matches[p1] == matches[p2]).values.reshape(-1, 1)
            y = (matches['Result'] == matches[p1]).astype(int)

            if len(np.unique(y)) > 1:
                model = LogisticRegression(random_state=42)
                model.fit(X, y)
                coefficient = model.coef_[0][0]
                normalized_coeff = 2 * (1 / (1 + np.exp(-coefficient)) - 0.5)
            else:
                win_rate = y.mean()
                normalized_coeff = (win_rate - 0.5) * 2

            pair_coefficients[(p1, p2)] = normalized_coeff

    return pair_coefficients


@app.route('/players', methods=['GET'])
def get_player_names():
    try:
        player_names = X.columns.tolist()
        return jsonify({'players': player_names}), 200
    except Exception as e:
        app.logger.error(f"Error in get_player_names: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/create_teams', methods=['POST'])
def create_teams():
    selected_players = sorted(request.json['players'])
    selected_coefficients = coefficients_df[coefficients_df['Player'].isin(selected_players)]
    pair_coefficients = calculate_pair_coefficients(selected_players)
    combinations_5 = list(combinations(selected_players, 5))

    max_individual_coeff = max(abs(coeff) for _, coeff in coefficients_list)
    max_pair_coeff = max(abs(coeff) for coeff in pair_coefficients.values())

    individual_weight = 0.8
    pair_weight = 0.2

    combination_scores = []
    for comb in combinations_5:
        individual_score = sum(selected_coefficients[selected_coefficients['Player'].isin(comb)]['Coefficient']) / max_individual_coeff
        pair_score = sum(pair_coefficients.get((p1, p2), 0) for p1, p2 in combinations(comb, 2)) / max_pair_coeff

        total_score = (individual_score * individual_weight) + (pair_score * pair_weight)
        total_score = round(total_score, 6)
        combination_scores.append((comb, total_score))

    combination_scores.sort(key=lambda x: (x[1], x[0]))

    best_comb_1, best_comb_2, min_diff = None, None, float('inf')

    for i in range(len(combination_scores)):
        for j in range(i + 1, len(combination_scores)):
            comb1 = set(combination_scores[i][0])
            comb2 = set(combination_scores[j][0])
            if not comb1.intersection(comb2):
                diff = abs(combination_scores[i][1] - combination_scores[j][1])
                if diff < min_diff or (diff == min_diff and combination_scores[i][0] < best_comb_1[0]):
                    min_diff = diff
                    best_comb_1 = combination_scores[i]
                    best_comb_2 = combination_scores[j]

    if best_comb_1 is None or best_comb_2 is None:
        return jsonify({
            "error": "Unable to create balanced teams with the selected players. Please try a different selection."
        }), 400

    avg_score = round((best_comb_1[1] + best_comb_2[1]) / 2, 6)

    result = {
        "team1": sorted(list(best_comb_1[0])),
        "team2": sorted(list(best_comb_2[0])),
        "team1_score": avg_score,
        "team2_score": avg_score
    }
    return jsonify(result)


@app.route('/update_match_result', methods=['POST'])
def update_match_result():
    try:
        data = request.json
        team1 = data['team1']
        team2 = data['team2']
        winner = data['winner']

        df = read_csv_from_github()

        new_row = pd.DataFrame(columns=df.columns)
        new_row.loc[0] = 0

        for player in team1:
            if player in new_row.columns:
                new_row.at[0, player] = 1
            else:
                return jsonify({"error": f"Player {player} not found in the dataset"}), 400

        for player in team2:
            if player in new_row.columns:
                new_row.at[0, player] = 2
            else:
                return jsonify({"error": f"Player {player} not found in the dataset"}), 400

        if 'Result' in new_row.columns:
            new_row.at[0, 'Result'] = 1 if winner == 'team1' else 2
        else:
            return jsonify({"error": "Result column not found in the dataset"}), 400

        df = pd.concat([df, new_row], ignore_index=True)

        save_csv_to_github(df)

        retrain_model()

        return jsonify({"message": "Match result updated and model retrained successfully"}), 200
    except Exception as e:
        app.logger.error(f"Error in update_match_result: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route('/sorted_coefficients', methods=['GET'])
def get_sorted_coefficients():
    sorted_coeffs = coefficients_df.sort_values('Coefficient', ascending=False)
    coeffs_list = sorted_coeffs.to_dict('records')
    return jsonify(coeffs_list)


@app.route('/pair_coefficients', methods=['GET'])
def get_pair_coefficients():
    players = X.columns.tolist()
    pair_coeffs = calculate_pair_coefficients(players)
    pair_coeffs_list = [{"player1": p1, "player2": p2, "coefficient": coeff} for (p1, p2), coeff in pair_coeffs.items()]
    pair_coeffs_list.sort(key=lambda x: x['coefficient'], reverse=True)
    return jsonify(pair_coeffs_list)


@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        retrain_model()
        return jsonify({"message": "Model retrained successfully"}), 200
    except Exception as e:
        app.logger.error(f"Error in retraining: {str(e)}")
        return jsonify({"error": f"An error occurred during retraining: {str(e)}"}), 500


@app.route('/create_teams_with_captains', methods=['POST'])
def create_teams_with_captains():
    data = request.json
    captain1 = data['captain1']
    captain2 = data['captain2']
    remaining_players = sorted(data['remaining_players'])

    if len(remaining_players) != 8:
        return jsonify({"error": "There must be exactly 8 remaining players"}), 400

    all_players = [captain1, captain2] + remaining_players
    all_coefficients = coefficients_df[coefficients_df['Player'].isin(all_players)]
    pair_coefficients = calculate_pair_coefficients(all_players)
    combinations_4 = list(combinations(remaining_players, 4))

    combination_scores = []
    for comb in combinations_4:
        team1 = tuple(sorted([captain1] + list(comb)))
        team2 = tuple(sorted([captain2] + list(set(remaining_players) - set(comb))))

        team1_score = sum(all_coefficients[all_coefficients['Player'].isin(team1)]['Coefficient'])
        team2_score = sum(all_coefficients[all_coefficients['Player'].isin(team2)]['Coefficient'])

        team1_score += sum(pair_coefficients.get((p1, p2), 0) for p1, p2 in combinations(team1, 2))
        team2_score += sum(pair_coefficients.get((p1, p2), 0) for p1, p2 in combinations(team2, 2))

        team1_score = round(team1_score, 6)
        team2_score = round(team2_score, 6)
        score_difference = abs(team1_score - team2_score)

        combination_scores.append((team1, team2, team1_score, team2_score, score_difference))

    combination_scores.sort(key=lambda x: (x[4], x[0], x[1]))

    best_combination = combination_scores[0]
    avg_score = round((best_combination[2] + best_combination[3]) / 2, 6)

    result = {
        "team1": list(best_combination[0]),
        "team2": list(best_combination[1]),
        "team1_score": avg_score,
        "team2_score": avg_score
    }
    return jsonify(result)


@app.route('/')
def serve_frontend():
    return send_from_directory(os.getcwd(), 'index.html')


@app.route('/player')
def index2():
    return send_from_directory(os.getcwd(), 'index2.html')


@app.route('/pair')
def index3():
    return send_from_directory(os.getcwd(), 'index3.html')


@app.route('/captains')
def index4():
    return send_from_directory(os.getcwd(), 'index4.html')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 8001)))
