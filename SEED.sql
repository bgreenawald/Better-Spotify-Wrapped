-- Placeholder seed data: this file ships generic example users only.
-- Keep real seed data (actual user ids/names) in a local, untracked file.
-- Requires the schema from DDL.sql (dim_users is declared there).

INSERT INTO dim_users (user_id, display_name, country) VALUES
('user_one', 'Example User', 'US'),
('user_two', 'Second User', 'US'),
('user_three', 'Third User', 'US'),
('user_four', 'Fourth User', 'US')
;
