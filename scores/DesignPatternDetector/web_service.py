import os
from datetime import datetime
import multiprocessing
import psycopg
import json
from web_app import get_github
db_connection = os.environ.get("DB_CONN")

class ScoreProcessor:
  running_score = False
  queue = multiprocessing.Queue()
  conn = psycopg.connect(db_connection)

  def enqueue_score(self, repo_url, score_id, branch, verbose, dataset):
    print("Enqueueing score")
    self.queue.put([repo_url, score_id, branch, verbose, dataset])

  def listen(self):
    while True:
      if not self.running_score and not self.queue.empty():
        repo_url, score_id, branch, verbose, dataset = self.queue.get()
        print("Processing score")
        self.running_score = True
        print(repo_url, score_id, branch, verbose, dataset)
        self.conn.cursor().execute("UPDATE scores.user_score SET score_status = 'CALCULATING', updated_at = %s WHERE id = %s", (datetime.now(), score_id))
        self.conn.commit()
        try:
          design_patterns = get_github(repo_url, branch, verbose, dataset)
          self.conn.cursor().execute("UPDATE scores.user_score SET result = %s, meta='{}', updated_at = %s, score_status = 'SUCCESS' WHERE id = %s", (json.dumps(design_patterns), datetime.now(), score_id))
        except Exception as e:
          self.conn.cursor().execute("UPDATE scores.user_score SET meta = %s, updated_at = %s, score_status = 'ERROR' WHERE id = %s", (json.dumps({"error": str(e)}), datetime.now(), score_id))
        self.conn.commit()
        self.running_score = False