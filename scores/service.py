import os
from datetime import datetime
import multiprocessing
import psycopg
import json
from app import One_Class_To_Rule_Them_All
db_connection = os.environ.get("DB_CONN")

scores = One_Class_To_Rule_Them_All()

class ScoreProcessor:
  running_score = False
  queue = multiprocessing.Queue()
  conn = psycopg.connect(db_connection)

  def enqueue_score(self, repo_url, score_id, branch):
    print("Enqueueing score")
    self.queue.put([repo_url, score_id, branch])

  def listen(self):
    while True:
      if not self.running_score and not self.queue.empty():
        repo_url, score_id, branch = self.queue.get()
        print("Processing score")
        self.running_score = True
        print(repo_url, score_id, branch)
        self.conn.cursor().execute("UPDATE scores.user_score SET score_status = 'CALCULATING', modified_on = %s WHERE id = %s", (datetime.now(), score_id))
        self.conn.commit()
        try:
          design_patterns = scores.designpatterns_score(repo_url, branch)
          self.conn.cursor().execute("UPDATE scores.user_score SET results = %s, modified_on = %s, score_status = 'SUCCESS' WHERE id = %s", (json.dumps(design_patterns), datetime.now(), score_id))
        except Exception as e:
          self.conn.cursor().execute("UPDATE scores.user_score SET meta = %s, modified_on = %s, score_status = 'ERROR' WHERE id = %s", (json.dumps({"error": str(e)}), datetime.now(), score_id))
        self.conn.commit()
        self.running_score = False