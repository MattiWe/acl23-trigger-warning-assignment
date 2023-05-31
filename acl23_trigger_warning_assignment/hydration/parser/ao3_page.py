class Ao3PageError(Exception):

    def __init__(self, error_name, description):
        super().__init__(f"[{error_name}] {description}")
        self.name = error_name
        self.description = description


# sometimes pages display errors in the html page but the response doesn't contain error codes
def check_for_errors(parser):
    # If the login box is displayed, the work is private
    login_elem = parser.query_selector("div#inner.wrapper > #main > #signin")
    if login_elem is not None:
        raise Ao3PageError("Work is private", "This work is private and needs a logged-in client to see.")
    error_elem = parser.query_selector("div#inner.wrapper > .system.errors")
    if error_elem is not None:
        error_name = error_elem.query_selector("h2.heading").text
        raise Ao3PageError(error_name, "error element in html.")


