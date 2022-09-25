
functions {
    real compute_log_likelihood(vector cur_skills, int n_per_race) {
    
        real cur_lik = 0;

        for (cur_position in 1:(n_per_race - 1)) {
            vector[n_per_race - cur_position + 1] other_skills;

            for (cur_other_position in cur_position:n_per_race) {
                other_skills[cur_other_position - cur_position + 1] = cur_skills[cur_other_position];
            }

            real cur_numerator = cur_skills[cur_position];
            real cur_denominator = log_sum_exp(other_skills);
            cur_lik += cur_numerator - cur_denominator;
        }
        
        return cur_lik;

    }
}
data {
    int n_drivers;
    int n_races;
    int n_teams;
    int n_finished_by_race[n_races];
    int n_finished_total;
    int n_seasons;
    
    int driver_placings[n_finished_total];
    int team_ids[n_finished_total];
    int season_id[n_finished_total];
    
    int n_dnf;
    
    int team_ids_dnf[n_dnf];
    int driver_ids_dnf[n_dnf];
    int season_ids_dnf[n_dnf];
    int race_ids_dnf[n_dnf];
    
    int new_reg_index;
}
parameters {
    vector[n_drivers] driver_init_raw;
    matrix[n_drivers, n_seasons - 1] driver_walk_raw;
    
    vector[n_drivers] driver_risk_init_raw;
    real<lower=0> driver_risk_init_sd;
        
    vector[n_teams] team_risk_init_raw;
    real<lower=0> team_risk_init_sd;

    matrix[n_teams, n_seasons - 1] team_risk_walk_raw;
    real<lower=0> team_risk_walk_sd;
    real<lower=0> new_reg_team_risk_extra_sd;
    
    real dnf_intercept;
    
    real<lower=0> driver_init_sd;
    real<lower=0> driver_season_sd;
    
    vector[n_teams] team_init_raw;
    matrix[n_teams, n_seasons - 1] team_walk_raw;
    real<lower=0> new_reg_team_walk_extra_sd;
    
    real<lower=0> team_init_sd;
    real<lower=0> team_season_sd;
}
transformed parameters {
    matrix[n_drivers, n_seasons] driver_skills;
    matrix[n_teams, n_seasons] team_skills;
    
    vector[n_drivers] driver_risk;
    matrix[n_teams, n_seasons] team_risk;
    
    vector[n_seasons - 1] team_season_sds = rep_vector(team_season_sd, n_seasons - 1);
    team_season_sds[new_reg_index - 1] += new_reg_team_walk_extra_sd;
    
    vector[n_seasons - 1] team_risk_walk_sds = rep_vector(team_risk_walk_sd, n_seasons - 1);
    team_risk_walk_sds[new_reg_index - 1] += new_reg_team_risk_extra_sd;
    
    for (cur_driver in 1:n_drivers) {
        vector[n_seasons - 1] cur_offsets = cumulative_sum(driver_walk_raw[cur_driver])' * driver_season_sd;
        driver_skills[cur_driver, 1] = driver_init_raw[cur_driver] * driver_init_sd;
        driver_skills[cur_driver, 2:n_seasons] = driver_skills[cur_driver, 1] + cur_offsets';
        
        // DNF risk
        driver_risk[cur_driver] = driver_risk_init_raw[cur_driver] * driver_risk_init_sd;
    }

    for (cur_team in 1:n_teams) {
        vector[n_seasons - 1] cur_offsets = cumulative_sum(team_walk_raw[cur_team])' .* team_season_sds;
        team_skills[cur_team, 1] = team_init_raw[cur_team] * team_init_sd;
        team_skills[cur_team, 2:n_seasons] = team_skills[cur_team, 1] + cur_offsets';
        
        // DNF risk
        cur_offsets = cumulative_sum(team_risk_walk_raw[cur_team])' .* team_risk_walk_sds;
        team_risk[cur_team, 1] = team_risk_init_raw[cur_team] * team_risk_init_sd;
        team_risk[cur_team, 2:n_seasons] = team_risk[cur_team, 1] + cur_offsets';
    }

}
model {
    int cur_start_index = 1;
    
    dnf_intercept ~ normal(0, 1);
    
    driver_risk_init_sd ~ normal(0, 1);
    driver_risk_init_raw ~ std_normal();
    
    new_reg_team_walk_extra_sd ~ normal(0, 1);
    new_reg_team_risk_extra_sd ~ normal(0, 1);
    
    team_risk_init_raw ~ std_normal();
    to_vector(team_risk_walk_raw) ~ std_normal();
    team_risk_init_sd ~ normal(0, 1);
    team_risk_walk_sd ~ normal(0, 1);
    
    driver_init_raw ~ std_normal();
    to_vector(driver_walk_raw) ~ std_normal();
    
    team_init_raw ~ std_normal();
    to_vector(team_walk_raw) ~ std_normal();
    
    team_init_sd ~ normal(0, 1);
    driver_init_sd ~ normal(0, 1);
    
    team_season_sd ~ normal(0, 1);
    driver_season_sd ~ normal(0, 1);
    
    // Conditional on finishing
    for (cur_race in 1:n_races) {
    
        int cur_finished = n_finished_by_race[cur_race];
    
        vector[cur_finished] cur_skills;
        
        int cur_placements[cur_finished] = driver_placings[cur_start_index:cur_start_index+cur_finished-1];
        int cur_teams[cur_finished] = team_ids[cur_start_index:cur_start_index+cur_finished-1];
        int cur_seasons[cur_finished] = season_id[cur_start_index:cur_start_index+cur_finished-1];
        
        for (i in 1:cur_finished) {
            cur_skills[i] = driver_skills[cur_placements[i], cur_seasons[i]] + 
            team_skills[cur_teams[i], cur_seasons[i]];
            0 ~ bernoulli_logit(driver_risk[cur_placements[i]] + 
            team_risk[cur_teams[i], cur_seasons[i]] + dnf_intercept);
        }
        
        target += compute_log_likelihood(cur_skills, cur_finished);
        
        cur_start_index += cur_finished;
        
    }
    
    // Conditional on not finishing
    for (cur_dnf in 1:n_dnf) {
        real cur_logit_prob_dnf = driver_risk[driver_ids_dnf[cur_dnf]] + 
        team_risk[team_ids_dnf[cur_dnf], season_ids_dnf[cur_dnf]] 
            + dnf_intercept;
        1 ~ bernoulli_logit(cur_logit_prob_dnf);
    }

}
generated quantities {

    int cur_start_index = 1;

    vector[n_races] log_likelihood;    
    
    // Conditional on finishing:
    for (cur_race in 1:n_races) {
    
        int cur_finished = n_finished_by_race[cur_race];
    
        vector[cur_finished] cur_skills;
        
        int cur_placements[cur_finished] = driver_placings[cur_start_index:cur_start_index+cur_finished-1];
        int cur_teams[cur_finished] = team_ids[cur_start_index:cur_start_index+cur_finished-1];
        int cur_seasons[cur_finished] = season_id[cur_start_index:cur_start_index+cur_finished-1];
        
        log_likelihood[cur_race] = 0;
        
        for (i in 1:cur_finished) {
            cur_skills[i] = driver_skills[cur_placements[i], cur_seasons[i]] + 
            team_skills[cur_teams[i], cur_seasons[i]];
            
            log_likelihood[cur_race] += bernoulli_logit_lpmf(
                0 | driver_risk[cur_placements[i]] + 
                team_risk[cur_teams[i], cur_seasons[i]] + dnf_intercept);
        }
        
        log_likelihood[cur_race] += compute_log_likelihood(cur_skills, cur_finished);
        
        cur_start_index += cur_finished;
        
    }
    
    // Conditional on not finishing:
    for (cur_dnf in 1:n_dnf) {
        real cur_logit_prob_dnf = driver_risk[driver_ids_dnf[cur_dnf]] + 
           team_risk[team_ids_dnf[cur_dnf], season_ids_dnf[cur_dnf]] + dnf_intercept;
        log_likelihood[race_ids_dnf[cur_dnf]] += bernoulli_logit_lpmf(1 |cur_logit_prob_dnf);
    }    
}

