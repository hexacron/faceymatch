"""The watch helper: match faces in any window of the operator's own machine (spec 6.11).

An operator-started desktop application, not part of the server. It captures one chosen
window, display or dragged region, posts each frame to `POST /api/live/match`, and draws the
returned boxes on a click-through overlay over the target.

It imports nothing from `app`. Everything it knows about the system it learns over loopback
HTTP from the running backend, which is what keeps it unable to touch the database, the model
sessions or the audit chain directly: every rule those paths enforce still applies, because
the helper only ever asks the endpoints that enforce them.
"""
