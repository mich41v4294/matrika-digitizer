use once_cell::sync::Lazy;
use rusqlite::Connection;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;
use std::time::Duration;

const MAX_POOL_CONNS_PER_PATH: usize = 12;
const MAX_POOLED_PATHS: usize = 24;

static CONNECTION_POOL: Lazy<Mutex<HashMap<String, Vec<Connection>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub struct PooledConnection {
    db_path: String,
    conn: Option<Connection>,
}

impl Deref for PooledConnection {
    type Target = Connection;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.conn.as_ref().expect("PooledConnection inner Connection missing")
    }
}

impl DerefMut for PooledConnection {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.conn.as_mut().expect("PooledConnection inner Connection missing")
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            if self.db_path.is_empty()
                || self.db_path == ":memory:"
                || self.db_path.starts_with("file::memory:")
            {
                return;
            }
            if !conn.is_autocommit() {
                let _ = conn.execute_batch("ROLLBACK;");
            }
            if let Ok(mut pool_map) = CONNECTION_POOL.lock() {
                if !pool_map.contains_key(&self.db_path) && pool_map.len() >= MAX_POOLED_PATHS {
                    pool_map.clear();
                }
                let bucket = pool_map.entry(self.db_path.clone()).or_default();
                if bucket.len() < MAX_POOL_CONNS_PER_PATH {
                    bucket.push(conn);
                }
            }
        }
    }
}

fn configure_new_connection(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.busy_timeout(Duration::from_secs(30))?;
    conn.set_prepared_statement_cache_capacity(128);
    let _ = conn.execute_batch(
        "PRAGMA synchronous=NORMAL;
         PRAGMA temp_store=MEMORY;
         PRAGMA cache_size=-64000;
         PRAGMA mmap_size=0;",
    );
    Ok(())
}

pub fn open_sqlite_pooled(db_path: &str) -> Result<PooledConnection, rusqlite::Error> {
    if !db_path.is_empty() && db_path != ":memory:" && !db_path.starts_with("file::memory:") {
        if let Ok(mut pool_map) = CONNECTION_POOL.lock() {
            if let Some(bucket) = pool_map.get_mut(db_path) {
                while let Some(conn) = bucket.pop() {
                    if conn.is_autocommit() {
                        return Ok(PooledConnection {
                            db_path: db_path.to_string(),
                            conn: Some(conn),
                        });
                    }
                }
            }
        }
    }

    let conn = Connection::open(db_path)?;
    configure_new_connection(&conn)?;
    Ok(PooledConnection {
        db_path: db_path.to_string(),
        conn: Some(conn),
    })
}

pub fn clear_connection_pool(db_path: Option<&str>) {
    if let Ok(mut pool_map) = CONNECTION_POOL.lock() {
        if let Some(p) = db_path {
            pool_map.remove(p);
        } else {
            pool_map.clear();
        }
    }
}
