CREATE TABLE dim_users (
  user_id         TEXT PRIMARY KEY,      -- internal UUID
  display_name    TEXT,
  country         TEXT,
  created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO dim_users (user_id, display_name, country) VALUES
('ben.greenawald', 'Ben Greenawald', 'US'),
('margo.greenawald', 'Margo Greenawald', 'US'),
('erin.greenawald', 'Erin Greenawald', 'US'),
('robby.woo', 'Robby Woo', 'US')
;